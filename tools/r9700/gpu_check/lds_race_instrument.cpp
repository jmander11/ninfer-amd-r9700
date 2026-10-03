// Instruments optimized gfx1201 device bitcode for the LDS race checker.
//
// Every load, store, and memory intrinsic that can address LDS gets a call into lds_race_rt.hip;
// every workgroup barrier (s.barrier, or s.barrier.wait of the gfx12 split form) advances the
// wave's barrier epoch. Kernels with no LDS access are left unchanged. The input must be optimized
// bitcode: a non-kernel function that still touches LDS or a barrier after inlining is rejected,
// because the epoch lives in the kernel's frame.
//
// usage: ninfer_lds_race_instrument <in.bc> <out.bc>

#include <llvm-c/BitReader.h>
#include <llvm-c/BitWriter.h>
#include <llvm-c/Core.h>
#include <llvm-c/Target.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "lds_race_layout.h"

namespace {

constexpr unsigned kAmdgpuKernelCallConv = 91;
constexpr unsigned kLds                  = 3;
constexpr unsigned kFlat                 = 0;
constexpr unsigned kPrivate              = 5;

struct Site {
    LLVMValueRef inst;
    LLVMValueRef pointer;
    LLVMValueRef bytes; // i32 value
    bool write;
};

std::string called_name(LLVMValueRef call) {
    LLVMValueRef callee = LLVMGetCalledValue(call);
    if (callee == nullptr || LLVMIsAFunction(callee) == nullptr) { return {}; }
    size_t length    = 0;
    const char* name = LLVMGetValueName2(callee, &length);
    return std::string(name, length);
}

bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

bool is_barrier(const std::string& name) {
    return name == "llvm.amdgcn.s.barrier" || name == "llvm.amdgcn.s.barrier.wait";
}

// ds_swizzle/ds_permute/ds_bpermute move values between lanes through the LDS crossbar without
// touching LDS memory.
bool is_cross_lane(const std::string& name) {
    return starts_with(name, "llvm.amdgcn.ds.swizzle") ||
           starts_with(name, "llvm.amdgcn.ds.permute") ||
           starts_with(name, "llvm.amdgcn.ds.bpermute");
}

unsigned pointer_space(LLVMValueRef pointer) {
    return LLVMGetPointerAddressSpace(LLVMTypeOf(pointer));
}

class Instrumenter {
public:
    explicit Instrumenter(LLVMModuleRef module)
        : module_(module), context_(LLVMGetModuleContext(module)),
          data_(LLVMGetModuleDataLayout(module)), builder_(LLVMCreateBuilderInContext(context_)) {
        i32_      = LLVMInt32TypeInContext(context_);
        void_     = LLVMVoidTypeInContext(context_);
        flat_ptr_ = LLVMPointerTypeInContext(context_, kFlat);
        lds_ptr_  = LLVMPointerTypeInContext(context_, kLds);
        // Opaque storage for the runtime's NinferRcState (lds_race_rt.hip asserts it fits).
        state_type_ = LLVMArrayType2(LLVMInt64TypeInContext(context_), ninfer::gpu_check::kStateBytes / 8);

        LLVMTypeRef enter_params[]   = {flat_ptr_, lds_ptr_};
        enter_type_                  = LLVMFunctionType(void_, enter_params, 2, 0);
        LLVMTypeRef barrier_params[] = {flat_ptr_};
        barrier_type_                = LLVMFunctionType(void_, barrier_params, 1, 0);
        LLVMTypeRef lds_params[]     = {flat_ptr_, lds_ptr_, i32_, i32_, flat_ptr_};
        lds_type_                    = LLVMFunctionType(void_, lds_params, 5, 0);
        LLVMTypeRef flat_params[]    = {flat_ptr_, flat_ptr_, i32_, i32_, flat_ptr_};
        flat_type_                   = LLVMFunctionType(void_, flat_params, 5, 0);
    }

    ~Instrumenter() { LLVMDisposeBuilder(builder_); }

    bool run() {
        bool ok = true;
        // The bitcode carries the AMDGPU attributor's "amdgpu-no-*" inferences from the first
        // optimization (no dispatch pointer, no workgroup id y, ...). The runtime calls added here
        // invalidate them; the final compile infers them again.
        for (LLVMValueRef fn = LLVMGetFirstFunction(module_); fn != nullptr;
             fn              = LLVMGetNextFunction(fn)) {
            strip_no_implicit_inputs(fn);
        }
        for (LLVMValueRef fn = LLVMGetFirstFunction(module_); fn != nullptr;
             fn              = LLVMGetNextFunction(fn)) {
            if (LLVMCountBasicBlocks(fn) == 0) { continue; }
            std::vector<Site> sites;
            std::vector<LLVMValueRef> barriers;
            collect(fn, sites, barriers);
            if (sites.empty() && barriers.empty()) { continue; }
            if (LLVMGetFunctionCallConv(fn) != kAmdgpuKernelCallConv) {
                size_t length    = 0;
                const char* name = LLVMGetValueName2(fn, &length);
                if (sites_touch_lds_statically(sites) || !barriers.empty()) {
                    std::fprintf(
                        stderr,
                        "lds_race_instrument: non-kernel function %.*s still accesses LDS or "
                        "a barrier after inlining\n",
                        static_cast<int>(length), name);
                    ok = false;
                }
                continue;
            }
            if (!sites_touch_lds_statically(sites) && !has_flat(sites)) {
                continue; // only barriers: nothing to check
            }
            instrument_kernel(fn, sites, barriers);
            ++kernels_;
        }
        return ok;
    }

    unsigned kernels() const { return kernels_; }

    unsigned accesses() const { return accesses_; }

private:
    static void strip_no_implicit_inputs(LLVMValueRef fn) {
        const LLVMAttributeIndex index = LLVMAttributeFunctionIndex;
        const unsigned count           = LLVMGetAttributeCountAtIndex(fn, index);
        std::vector<LLVMAttributeRef> attributes(count);
        LLVMGetAttributesAtIndex(fn, index, attributes.data());
        std::vector<std::string> keys;
        for (LLVMAttributeRef attribute : attributes) {
            if (LLVMIsStringAttribute(attribute) == 0) { continue; }
            unsigned length = 0;
            const char* key = LLVMGetStringAttributeKind(attribute, &length);
            const std::string name(key, length);
            if (starts_with(name, "amdgpu-no-")) { keys.push_back(name); }
        }
        for (const std::string& key : keys) {
            LLVMRemoveStringAttributeAtIndex(fn, index, key.c_str(),
                                             static_cast<unsigned>(key.size()));
        }
    }

    static bool sites_touch_lds_statically(const std::vector<Site>& sites) {
        for (const Site& s : sites) {
            if (pointer_space(s.pointer) == kLds) { return true; }
        }
        return false;
    }

    static bool has_flat(const std::vector<Site>& sites) {
        for (const Site& s : sites) {
            if (pointer_space(s.pointer) == kFlat) { return true; }
        }
        return false;
    }

    LLVMValueRef bytes_of(LLVMTypeRef type) {
        return LLVMConstInt(i32_, LLVMStoreSizeOfType(data_, type), 0);
    }

    void add_site(std::vector<Site>& sites, LLVMValueRef inst, LLVMValueRef pointer,
                  LLVMValueRef bytes, bool write) {
        const unsigned space = pointer_space(pointer);
        if (space == kLds || space == kFlat) { sites.push_back({inst, pointer, bytes, write}); }
    }

    void collect(LLVMValueRef fn, std::vector<Site>& sites, std::vector<LLVMValueRef>& barriers) {
        for (LLVMBasicBlockRef bb = LLVMGetFirstBasicBlock(fn); bb != nullptr;
             bb                   = LLVMGetNextBasicBlock(bb)) {
            for (LLVMValueRef inst = LLVMGetFirstInstruction(bb); inst != nullptr;
                 inst              = LLVMGetNextInstruction(inst)) {
                switch (LLVMGetInstructionOpcode(inst)) {
                case LLVMLoad:
                    if (LLVMGetOrdering(inst) == LLVMAtomicOrderingNotAtomic) {
                        add_site(sites, inst, LLVMGetOperand(inst, 0), bytes_of(LLVMTypeOf(inst)),
                                 false);
                    }
                    break;
                case LLVMStore:
                    if (LLVMGetOrdering(inst) == LLVMAtomicOrderingNotAtomic) {
                        add_site(sites, inst, LLVMGetOperand(inst, 1),
                                 bytes_of(LLVMTypeOf(LLVMGetOperand(inst, 0))), true);
                    }
                    break;
                case LLVMCall: {
                    const std::string name = called_name(inst);
                    if (is_barrier(name)) {
                        barriers.push_back(inst);
                    } else if (starts_with(name, "llvm.memcpy.") ||
                               starts_with(name, "llvm.memmove.")) {
                        LLVMValueRef bytes = LLVMBuildTruncOrBitCast(
                            builder_at(inst), LLVMGetOperand(inst, 2), i32_, "");
                        add_site(sites, inst, LLVMGetOperand(inst, 0), bytes, true);
                        add_site(sites, inst, LLVMGetOperand(inst, 1), bytes, false);
                    } else if (starts_with(name, "llvm.memset.")) {
                        LLVMValueRef bytes = LLVMBuildTruncOrBitCast(
                            builder_at(inst), LLVMGetOperand(inst, 2), i32_, "");
                        add_site(sites, inst, LLVMGetOperand(inst, 0), bytes, true);
                    } else if ((starts_with(name, "llvm.amdgcn.ds.") && !is_cross_lane(name)) ||
                               starts_with(name, "llvm.amdgcn.global.load.lds") ||
                               starts_with(name, "llvm.amdgcn.load.to.lds") ||
                               starts_with(name, "llvm.amdgcn.raw.buffer.load.lds") ||
                               starts_with(name, "llvm.amdgcn.struct.buffer.load.lds")) {
                        std::fprintf(stderr, "lds_race_instrument: unhandled LDS intrinsic %s\n",
                                     name.c_str());
                    }
                    break;
                }
                default:
                    // atomicrmw/cmpxchg (like atomic loads and stores above) are synchronization,
                    // not checked accesses.
                    break;
                }
            }
        }
    }

    LLVMBuilderRef builder_at(LLVMValueRef inst) {
        LLVMPositionBuilderBefore(builder_, inst);
        return builder_;
    }

    LLVMValueRef site_string(LLVMValueRef fn, LLVMValueRef inst) {
        // Source location first, then the (mangled) kernel name; the device runtime truncates.
        std::string text;
        unsigned file_length = 0;
        const char* file     = LLVMGetDebugLocFilename(inst, &file_length);
        if (file != nullptr && file_length > 0) {
            std::string path(file, file_length);
            const size_t slash = path.rfind('/');
            text               = slash == std::string::npos ? path : path.substr(slash + 1);
            text += ":" + std::to_string(LLVMGetDebugLocLine(inst)) + " ";
        }
        size_t fn_length    = 0;
        const char* fn_name = LLVMGetValueName2(fn, &fn_length);
        text.append(fn_name, fn_length);
        LLVMValueRef value = LLVMConstStringInContext2(context_, text.c_str(), text.size(), 0);
        LLVMValueRef global =
            LLVMAddGlobalInAddressSpace(module_, LLVMTypeOf(value), "__ninfer_rc_site", 4);
        LLVMSetInitializer(global, value);
        LLVMSetGlobalConstant(global, 1);
        LLVMSetLinkage(global, LLVMPrivateLinkage);
        LLVMSetUnnamedAddress(global, LLVMGlobalUnnamedAddr);
        return LLVMConstAddrSpaceCast(global, flat_ptr_);
    }

    LLVMValueRef declare(const char* name, LLVMTypeRef type) {
        LLVMValueRef fn = LLVMGetNamedFunction(module_, name);
        return fn != nullptr ? fn : LLVMAddFunction(module_, name, type);
    }

    void instrument_kernel(LLVMValueRef fn, const std::vector<Site>& sites,
                           const std::vector<LLVMValueRef>& barriers) {
        LLVMValueRef enter   = declare("__ninfer_rc_enter", enter_type_);
        LLVMValueRef barrier = declare("__ninfer_rc_barrier", barrier_type_);
        LLVMValueRef lds     = declare("__ninfer_rc_lds", lds_type_);
        LLVMValueRef flat    = declare("__ninfer_rc_flat", flat_type_);

        LLVMValueRef nonce = LLVMAddGlobalInAddressSpace(module_, i32_, "__ninfer_rc_nonce", kLds);
        LLVMSetInitializer(nonce, LLVMGetPoison(i32_));
        LLVMSetLinkage(nonce, LLVMInternalLinkage);
        LLVMSetAlignment(nonce, 4);

        LLVMBasicBlockRef entry = LLVMGetEntryBasicBlock(fn);
        LLVMValueRef first      = LLVMGetFirstInstruction(entry);
        while (first != nullptr && LLVMGetInstructionOpcode(first) == LLVMAlloca) {
            first = LLVMGetNextInstruction(first);
        }
        LLVMPositionBuilderBefore(builder_, first);
        LLVMValueRef state_private = LLVMBuildAlloca(builder_, state_type_, "rc.state");
        LLVMValueRef state =
            LLVMBuildAddrSpaceCast(builder_, state_private, flat_ptr_, "rc.state.flat");
        LLVMValueRef enter_args[] = {state, nonce};
        LLVMBuildCall2(builder_, enter_type_, enter, enter_args, 2, "");

        for (const Site& s : sites) {
            LLVMPositionBuilderBefore(builder_, s.inst);
            LLVMValueRef write = LLVMConstInt(i32_, s.write ? 1 : 0, 0);
            LLVMValueRef where = site_string(fn, s.inst);
            if (pointer_space(s.pointer) == kLds) {
                LLVMValueRef args[] = {state, s.pointer, s.bytes, write, where};
                LLVMBuildCall2(builder_, lds_type_, lds, args, 5, "");
            } else {
                LLVMValueRef args[] = {state, s.pointer, s.bytes, write, where};
                LLVMBuildCall2(builder_, flat_type_, flat, args, 5, "");
            }
            ++accesses_;
        }
        for (LLVMValueRef b : barriers) {
            LLVMValueRef next = LLVMGetNextInstruction(b);
            LLVMPositionBuilderBefore(builder_, next);
            LLVMValueRef args[] = {state};
            LLVMBuildCall2(builder_, barrier_type_, barrier, args, 1, "");
        }
        (void)kPrivate;
    }

    LLVMModuleRef module_;
    LLVMContextRef context_;
    LLVMTargetDataRef data_;
    LLVMBuilderRef builder_;
    LLVMTypeRef i32_{};
    LLVMTypeRef void_{};
    LLVMTypeRef flat_ptr_{};
    LLVMTypeRef lds_ptr_{};
    LLVMTypeRef state_type_{};
    LLVMTypeRef enter_type_{};
    LLVMTypeRef barrier_type_{};
    LLVMTypeRef lds_type_{};
    LLVMTypeRef flat_type_{};
    unsigned kernels_  = 0;
    unsigned accesses_ = 0;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <in.bc> <out.bc>\n", argv[0]);
        return 2;
    }
    LLVMContextRef context     = LLVMContextCreate();
    LLVMMemoryBufferRef buffer = nullptr;
    char* message              = nullptr;
    if (LLVMCreateMemoryBufferWithContentsOfFile(argv[1], &buffer, &message) != 0) {
        std::fprintf(stderr, "lds_race_instrument: %s\n", message);
        return 1;
    }
    LLVMModuleRef module = nullptr;
    if (LLVMParseBitcodeInContext2(context, buffer, &module) != 0) {
        std::fprintf(stderr, "lds_race_instrument: cannot parse %s\n", argv[1]);
        return 1;
    }
    LLVMDisposeMemoryBuffer(buffer);
    Instrumenter instrumenter(module);
    const bool ok = instrumenter.run();
    if (LLVMWriteBitcodeToFile(module, argv[2]) != 0) {
        std::fprintf(stderr, "lds_race_instrument: cannot write %s\n", argv[2]);
        return 1;
    }
    if (std::getenv("NINFER_RC_VERBOSE") != nullptr) {
        std::fprintf(stderr, "lds_race_instrument: %u kernels, %u accesses\n",
                     instrumenter.kernels(), instrumenter.accesses());
    }
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    return ok ? 0 : 1;
}

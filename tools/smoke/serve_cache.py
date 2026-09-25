"""Exercise a running cache-enabled server without overriding its temperature.

Run with Python3.11; this sends real inference requests. Cache persistence across
restart is qualified separately by ninfer_r9700_engine_cache_cancel_qual.
"""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080')
    args = parser.parse_args()

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.base_url.rstrip('/') + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    assert request('/health')['status'] == 'ok'
    models = request('/v1/models')['data']
    assert models
    model = models[0]['id']
    results = []
    # Revisit A after unrelated B to exercise retained-prefix lookup, not merely
    # repeated output sampling. Requests intentionally inherit server temperature.
    for prompt in ['List four colors.', 'List four animals.', 'List four colors.']:
        reply = request('/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 32, 'enable_thinking': False, 'seed': 42,
        })
        assert reply['choices'][0]['message']['content']
        assert reply['usage']['completion_tokens'] > 0
        details = reply['usage']['prompt_tokens_details']
        stats = details['ninfer']
        assert 'kv_ram' in stats and 'kv_disk' in stats, 'both cache tiers must be enabled'
        results.append({'cached_tokens': details['cached_tokens'], 'stats': stats})
    response = request('/v1/responses', {
        'model': model, 'input': 'Say hello.', 'max_output_tokens': 32,
        'reasoning': {'effort': 'none'},
    })
    assert response['output'] and response['usage']['output_tokens'] > 0
    message = request('/v1/messages', {
        'model': model, 'messages': [{'role': 'user', 'content': 'Say hello.'}],
        'max_tokens': 32,
    })
    assert message['content'] and message['usage']['output_tokens'] > 0
    payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Say hello.'}],
               'max_tokens': 32, 'enable_thinking': False, 'stream': True}
    req = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as stream:
        events = stream.read().decode()
    assert 'data: [DONE]' in events and 'chat.completion.chunk' in events
    print(json.dumps({'status': 'PASS', 'model': model,
                      'routes': ['chat', 'responses', 'messages', 'chat_sse'],
                      'chat_cache_observations': results}, indent=2))


if __name__ == '__main__':
    main()

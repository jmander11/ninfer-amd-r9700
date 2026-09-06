#!/usr/bin/env python3
"""Recompute the discriminator decision from retained raw reports and process records."""

import sys

import run

sys.argv = [sys.argv[0], "--analyze-only"]
raise SystemExit(run.main())

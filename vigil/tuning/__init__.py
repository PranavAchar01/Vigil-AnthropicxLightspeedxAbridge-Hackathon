"""Vigil closed-loop AI tuning.

A single reusable optimization loop (`optimizer.optimize`) driven by per-AI
evaluators (vision / audio / face / fusion / reasoning). Each evaluator scores a
candidate parameter set against a labeled benchmark; the loop searches the space,
keeps the incumbent best, refines locally, logs every trial, and promotes the
winning params to `config/tuned.env`, which the running system loads.

This is the "loop engineering technique": evaluate → search → promote → re-evaluate,
applied to every model and agent loop in the system.
"""

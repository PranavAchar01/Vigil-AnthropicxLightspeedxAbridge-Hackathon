"use client";

import { useCallback, useEffect, useState } from "react";
import {
  fetchVigilSession,
  localSession,
  type VigilRole,
  type VigilSession,
} from "./demoSession";
import { useVigilBackend } from "./lib/useVigilBackend";

export function useRoleSession(role: VigilRole, intervalMs = 2500) {
  const backend = useVigilBackend();
  const [session, setSession] = useState<VigilSession>(() => localSession(role, 0));
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      if (!backend) throw new Error("Backend unavailable");
      setSession(await fetchVigilSession(backend, role));
    } catch {
      setSession((current) => localSession(role, current.demo.step));
    } finally {
      setLoading(false);
    }
  }, [backend, role]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs, refresh]);

  return { backend, session, loading, refresh };
}

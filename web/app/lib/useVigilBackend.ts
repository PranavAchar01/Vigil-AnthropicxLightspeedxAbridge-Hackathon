"use client";

import { useEffect, useState } from "react";
import { supabase } from "./supabase";

const configuredBackend = (process.env.NEXT_PUBLIC_VIGIL_URL || "").replace(/\/$/, "");

function initialBackend(): string {
  if (configuredBackend) return configuredBackend;
  if (typeof window !== "undefined" && ["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    return "http://localhost:8000";
  }
  return "";
}

export function useVigilBackend(): string {
  const [backend, setBackend] = useState(initialBackend);

  useEffect(() => {
    const client = supabase;
    if (!client) return;

    let active = true;
    const apply = (url?: string | null) => {
      if (active && url) setBackend(url.replace(/\/$/, ""));
    };

    void client
      .from("vigil_runtime")
      .select("url")
      .eq("id", "backend")
      .maybeSingle()
      .then(({ data }) => apply((data as { url?: string } | null)?.url));

    const channel = client
      .channel("vigil_runtime_backend")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "vigil_runtime", filter: "id=eq.backend" },
        (payload) => apply((payload.new as { url?: string }).url),
      )
      .subscribe();

    return () => {
      active = false;
      void client.removeChannel(channel);
    };
  }, []);

  return backend;
}

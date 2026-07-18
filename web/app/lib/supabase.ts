import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY || "";

// Publishable (anon) key only — safe for the browser; reads are gated by RLS.
export const supabase = url && key ? createClient(url, key) : null;

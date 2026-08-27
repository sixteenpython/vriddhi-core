import { useEffect, useRef, useState } from "react";
import type { Campaign } from "./api";

export function RapidClock({ campaign, onExpire, disabled, forceActive = false }: { campaign: Campaign; onExpire: () => void; disabled: boolean; forceActive?: boolean }) {
  const active = forceActive || (campaign.mode === "RAPID" && campaign.status === "ACTIVE" && campaign.current_move > 1);
  const key = `bti-rapid-deadline-${campaign.campaign_id}-${campaign.current_move}`;
  const [remaining, setRemaining] = useState(120);
  const expireRef = useRef(onExpire);
  expireRef.current = onExpire;
  useEffect(() => {
    if (!active || disabled) return;
    let deadline = Number(localStorage.getItem(key));
    if (!deadline) {
      deadline = Date.now() + 120_000;
      localStorage.setItem(key, String(deadline));
    }
    let fired = false;
    const tick = () => {
      const seconds = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
      setRemaining(seconds);
      if (!seconds && !fired) {
        fired = true;
        expireRef.current();
      }
    };
    tick();
    const timer = window.setInterval(tick, 250);
    return () => window.clearInterval(timer);
  }, [active, disabled, key]);
  if (!active || disabled) return null;
  return <div className={`rapid-clock ${remaining <= 20 ? "urgent" : ""}`}><span>RAPID DECISION CLOCK</span><b>{String(Math.floor(remaining / 60)).padStart(2, "0")}:{String(remaining % 60).padStart(2, "0")}</b><small>At zero, BTI records HOLD and advances the market.</small></div>;
}

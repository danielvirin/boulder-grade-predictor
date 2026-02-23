import { useState, useCallback, useMemo, useRef, useEffect } from "react";

// --- Board geometry ---
const COLS = 17;
const ROWS = 19;
const HOLD_SPACING = 32;
const BOARD_PAD = 16;
const HOLD_R = 11;

// Generate hold grid matching real Kilter layout density
const genHolds = () => {
  const holds = [];
  let id = 1000;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const ox = r % 2 === 1 ? HOLD_SPACING * 0.5 : 0;
      holds.push({
        id: id++,
        r, c,
        x: BOARD_PAD + c * HOLD_SPACING + ox,
        y: BOARD_PAD + r * HOLD_SPACING,
      });
    }
  }
  return holds;
};
const HOLDS = genHolds();
const BW = BOARD_PAD * 2 + (COLS - 1) * HOLD_SPACING + HOLD_SPACING;
const BH = BOARD_PAD * 2 + (ROWS - 1) * HOLD_SPACING;

// Role config matching the real app colours
const ROLES = {
  start: { color: "#22c55e", glow: "#22c55e", label: "Start", code: "r12" },
  hand: { color: "#06d6e0", glow: "#06d6e0", label: "Hand", code: "r13" },
  finish: { color: "#e040a0", glow: "#e040a0", label: "Finish", code: "r14" },
  foot: { color: "#f59e0b", glow: "#f59e0b", label: "Foot", code: "r15" },
};
const ROLE_ORDER = ["start", "hand", "finish", "foot"];
const ROLE_FROM_CODE = { r12: "start", r13: "hand", r14: "finish", r15: "foot" };

const V_GRADES = ["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15"];
const GRADE_COLORS = {
  V0:"#4ade80",V1:"#86efac",V2:"#bef264",V3:"#fde047",V4:"#fdba74",V5:"#fb923c",
  V6:"#f87171",V7:"#ef4444",V8:"#dc2626",V9:"#b91c1c",V10:"#991b1b",V11:"#a855f7",
  V12:"#9333ea",V13:"#7c3aed",V14:"#6d28d9",V15:"#4c1d95",
};

// --- Sample climbs database ---
const CLIMBS = [
  { uuid:"c001", name:"Warm Up Slab", setter:"RouteBot", grade:"V0", angle:15, ascents:4820, quality:3.2, layout:"p1002r15p1003r15p1034r12p1035r12p1067r13p1098r13p1130r13p1162r14" },
  { uuid:"c002", name:"Jug Haul", setter:"CrimpKing", grade:"V1", angle:20, ascents:3910, quality:3.8, layout:"p1003r15p1034r12p1035r12p1068r13p1069r15p1101r13p1133r13p1165r14" },
  { uuid:"c003", name:"The Traverse", setter:"SetterSam", grade:"V2", angle:25, ascents:3200, quality:4.1, layout:"p1002r15p1003r15p1034r12p1036r12p1069r13p1070r13p1072r13p1104r13p1137r14" },
  { uuid:"c004", name:"Pinch Me", setter:"BoulderQueen", grade:"V3", angle:30, ascents:2850, quality:3.9, layout:"p1003r15p1035r12p1036r12p1068r13p1069r15p1100r13p1101r13p1132r13p1164r13p1196r14" },
  { uuid:"c005", name:"Crimp City", setter:"FingerStrength", grade:"V4", angle:35, ascents:2100, quality:4.3, layout:"p1002r15p1034r12p1035r12p1067r13p1098r15p1099r13p1131r13p1163r13p1195r13p1227r14" },
  { uuid:"c006", name:"Dyno Time", setter:"PowerClimber", grade:"V5", angle:40, ascents:1780, quality:4.5, layout:"p1003r15p1004r15p1035r12p1036r12p1100r13p1133r13p1165r15p1197r13p1229r14" },
  { uuid:"c007", name:"The Crux", setter:"ProSetter", grade:"V5", angle:40, ascents:2340, quality:4.0, layout:"p1002r15p1003r15p1034r12p1035r12p1067r13p1069r13p1100r13p1132r13p1133r15p1164r13p1196r14" },
  { uuid:"c008", name:"Sloper Madness", setter:"CrimpKing", grade:"V6", angle:40, ascents:1450, quality:4.2, layout:"p1003r15p1004r15p1036r12p1037r12p1069r13p1101r13p1102r13p1134r13p1166r13p1198r13p1230r14" },
  { uuid:"c009", name:"Power Endurance", setter:"RouteBot", grade:"V6", angle:45, ascents:1200, quality:3.7, layout:"p1002r15p1034r12p1035r12p1067r13p1068r15p1099r13p1100r13p1131r13p1163r13p1164r15p1195r13p1227r14" },
  { uuid:"c010", name:"Hammer Drop", setter:"VinceThePrince", grade:"V8", angle:40, ascents:3100, quality:4.7, layout:"p1003r15p1035r12p1036r12p1068r13p1100r13p1101r15p1133r13p1165r13p1197r13p1229r13p1261r14" },
  { uuid:"c011", name:"Compression Test", setter:"BoulderQueen", grade:"V7", angle:45, ascents:890, quality:4.4, layout:"p1002r15p1003r15p1034r12p1036r12p1068r13p1069r13p1101r13p1133r13p1134r15p1166r13p1198r13p1230r14" },
  { uuid:"c012", name:"Finger Buster", setter:"FingerStrength", grade:"V8", angle:45, ascents:670, quality:4.1, layout:"p1003r15p1035r12p1036r12p1068r13p1100r13p1132r13p1133r15p1164r13p1196r13p1228r13p1260r14" },
  { uuid:"c013", name:"The King Line", setter:"ProSetter", grade:"V9", angle:50, ascents:420, quality:4.8, layout:"p1002r15p1003r15p1035r12p1036r12p1068r13p1069r15p1100r13p1132r13p1164r13p1196r13p1228r13p1260r14" },
  { uuid:"c014", name:"Moonwalk", setter:"SetterSam", grade:"V4", angle:25, ascents:1950, quality:3.5, layout:"p1003r15p1034r12p1035r12p1067r13p1068r13p1070r13p1072r13p1104r13p1136r14" },
  { uuid:"c015", name:"The Eliminator", setter:"PowerClimber", grade:"V10", angle:50, ascents:280, quality:4.9, layout:"p1002r15p1003r15p1035r12p1036r12p1068r13p1100r13p1101r15p1132r13p1164r13p1196r13p1228r13p1229r15p1260r13p1261r14" },
  { uuid:"c016", name:"Campus Party", setter:"CrimpKing", grade:"V7", angle:45, ascents:1100, quality:4.0, layout:"p1003r15p1035r12p1036r12p1099r13p1131r13p1163r13p1195r13p1227r14" },
  { uuid:"c017", name:"Narasaki Bounce", setter:"VinceThePrince", grade:"V7", angle:40, ascents:2800, quality:4.6, layout:"p1002r15p1003r15p1035r12p1036r12p1068r13p1069r15p1101r13p1133r13p1165r13p1197r14" },
  { uuid:"c018", name:"Roof Rat", setter:"BoulderQueen", grade:"V9", angle:55, ascents:350, quality:4.3, layout:"p1003r15p1004r15p1036r12p1037r12p1069r13p1070r15p1101r13p1102r13p1134r13p1166r13p1198r13p1230r14" },
  { uuid:"c019", name:"Static Line", setter:"SetterSam", grade:"V3", angle:20, ascents:2600, quality:3.6, layout:"p1003r15p1035r12p1036r12p1068r13p1100r13p1132r13p1164r14" },
  { uuid:"c020", name:"Whip It", setter:"PowerClimber", grade:"V6", angle:35, ascents:1600, quality:4.1, layout:"p1002r15p1034r12p1035r12p1067r13p1099r13p1100r15p1131r13p1163r13p1164r15p1196r13p1228r14" },
  { uuid:"c021", name:"mini flowy lache compact", setter:"Daan_vo", grade:"V7", angle:40, ascents:1240, quality:3.0, layout:"p1005r15p1006r15p1039r12p1040r12p1073r13p1074r13p1106r13p1138r13p1170r14" },
];

const parseLayout = (layout) => {
  const pat = /p(\d+)r(\d+)/g;
  const h = {};
  let m;
  while ((m = pat.exec(layout))) {
    const pid = parseInt(m[1]);
    const role = ROLE_FROM_CODE["r" + m[2]] || "hand";
    const bh = HOLDS.find(x => x.id === pid);
    if (bh) h[bh.id] = role;
    else {
      const idx = Math.abs((pid - 1000) % HOLDS.length);
      if (HOLDS[idx]) h[HOLDS[idx].id] = role;
    }
  }
  return h;
};

// --- Grade estimation (angle-reactive) ---
const estimate = (sel, angle) => {
  const entries = Object.entries(sel).filter(([_, r]) => r !== "none");
  const holds = entries.map(([id]) => {
    const p = HOLDS.find(h => h.id === parseInt(id));
    return p ? { ...p, role: sel[id] } : null;
  }).filter(Boolean);
  if (holds.length < 3) return null;

  const hands = holds.filter(h => ["hand","start","finish"].includes(h.role));
  const feet = holds.filter(h => h.role === "foot");
  const xs = holds.map(h => h.x), ys = holds.map(h => h.y);
  const xSpan = Math.max(...xs) - Math.min(...xs);
  const ySpan = Math.max(...ys) - Math.min(...ys);

  const sorted = [...hands].sort((a, b) => a.y - b.y);
  let totalD = 0, maxD = 0;
  for (let i = 1; i < sorted.length; i++) {
    const d = Math.hypot(sorted[i].x - sorted[i-1].x, sorted[i].y - sorted[i-1].y);
    totalD += d;
    maxD = Math.max(maxD, d);
  }
  const avgD = sorted.length > 1 ? totalD / (sorted.length - 1) : 0;

  // Angle is the primary difficulty multiplier — steeper = exponentially harder
  const angleFactor = 1 + (angle / 70) * 1.8;
  let score = 0;
  score += angle * 0.1;
  score += Math.max(0, avgD - 30) * 0.05 * angleFactor;
  score += Math.max(0, maxD - 50) * 0.03 * angleFactor;
  score += Math.max(0, holds.length - 6) * 0.25;
  score -= feet.length * 0.5;
  score += xSpan * 0.008;
  score += Math.max(0, ySpan - 150) * 0.004;
  // Fewer footholds on steep = much harder
  if (angle > 30 && feet.length < 2) score += (angle - 30) * 0.04;

  const vGrade = Math.max(0, Math.min(15, Math.round(score)));
  const conf = Math.max(0.35, Math.min(0.92, 0.65 + holds.length * 0.025 - Math.abs(score - vGrade) * 0.2));

  return {
    grade: "V" + vGrade, numeric: vGrade, confidence: conf,
    features: {
      holds: holds.length, hands: hands.length, feet: feet.length,
      avgMove: Math.round(avgD), maxMove: Math.round(maxD),
      xSpan: Math.round(xSpan), ySpan: Math.round(ySpan), angle,
    },
  };
};

const getVerdict = (pred, assigned) => {
  if (!assigned) return null;
  const a = parseInt(assigned.replace("V", ""));
  const d = pred - a;
  if (Math.abs(d) < 1) return { emoji: "✅", text: "Grade looks accurate", color: "#22c55e" };
  if (d >= 2) return { emoji: "🔴", text: "Sandbagged — feels V" + pred, color: "#ef4444" };
  if (d >= 1) return { emoji: "🟡", text: "Slightly hard for grade", color: "#eab308" };
  if (d <= -2) return { emoji: "🔵", text: "Soft — feels V" + pred, color: "#3b82f6" };
  if (d <= -1) return { emoji: "🟡", text: "Slightly soft for grade", color: "#eab308" };
  return { emoji: "✅", text: "Accurate", color: "#22c55e" };
};

// --- Hold shapes for realistic look ---
const HoldShape = ({ x, y, seed }) => {
  // Deterministic pseudo-random shape variation
  const s = ((seed * 9301 + 49297) % 233280) / 233280;
  const s2 = ((seed * 7919 + 13337) % 177777) / 177777;
  const rot = s * 360;
  const scaleX = 0.7 + s2 * 0.5;
  const scaleY = 0.65 + s * 0.45;
  return (
    <ellipse
      cx={x} cy={y}
      rx={HOLD_R * scaleX} ry={HOLD_R * scaleY}
      transform={`rotate(${rot} ${x} ${y})`}
      fill="#2a2a2a"
      stroke="#333"
      strokeWidth={0.5}
    />
  );
};

const ActiveHold = ({ x, y, role, seed }) => {
  const cfg = ROLES[role];
  if (!cfg) return null;
  const s = ((seed * 9301 + 49297) % 233280) / 233280;
  const s2 = ((seed * 7919 + 13337) % 177777) / 177777;
  const rot = s * 360;
  const scaleX = 0.7 + s2 * 0.5;
  const scaleY = 0.65 + s * 0.45;
  return (
    <g>
      {/* Outer glow */}
      <ellipse cx={x} cy={y}
        rx={HOLD_R * scaleX + 6} ry={HOLD_R * scaleY + 6}
        transform={`rotate(${rot} ${x} ${y})`}
        fill="none" stroke={cfg.glow} strokeWidth={1}
        opacity={0.15} />
      {/* LED ring */}
      <ellipse cx={x} cy={y}
        rx={HOLD_R * scaleX + 3} ry={HOLD_R * scaleY + 3}
        transform={`rotate(${rot} ${x} ${y})`}
        fill="none"
        stroke={cfg.color}
        strokeWidth={2.5}
        opacity={0.9}
        style={{ filter: `drop-shadow(0 0 6px ${cfg.glow})` }}
      />
      {/* Hold body */}
      <ellipse cx={x} cy={y}
        rx={HOLD_R * scaleX} ry={HOLD_R * scaleY}
        transform={`rotate(${rot} ${x} ${y})`}
        fill="#2a2a2a"
        stroke="#333"
        strokeWidth={0.5}
      />
      {/* Subtle highlight */}
      <ellipse cx={x - 2} cy={y - 2}
        rx={HOLD_R * scaleX * 0.4} ry={HOLD_R * scaleY * 0.35}
        transform={`rotate(${rot} ${x} ${y})`}
        fill="rgba(255,255,255,0.06)"
      />
    </g>
  );
};

// --- Search component ---
const Search = ({ onSelect, selected }) => {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [grade, setGrade] = useState("all");
  const ref = useRef(null);

  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const results = useMemo(() => {
    let f = CLIMBS;
    if (q.trim()) {
      const lq = q.toLowerCase();
      f = f.filter(c => c.name.toLowerCase().includes(lq) || c.setter.toLowerCase().includes(lq) || c.grade.toLowerCase() === lq);
    }
    if (grade !== "all") f = f.filter(c => c.grade === grade);
    return f.sort((a, b) => b.ascents - a.ascents);
  }, [q, grade]);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <div style={{ display: "flex", gap: 5 }}>
        <div style={{ flex: 1, position: "relative" }}>
          <input value={q}
            onChange={e => { setQ(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            placeholder="Search by name, setter or grade..."
            style={{
              width: "100%", padding: "9px 10px 9px 30px",
              background: "#111", border: "1px solid #222", borderRadius: 8,
              color: "#ddd", fontSize: 12, outline: "none", boxSizing: "border-box",
              fontFamily: "inherit",
            }} />
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
            stroke="#444" strokeWidth="2.5" strokeLinecap="round"
            style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)" }}>
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
          </svg>
        </div>
        <select value={grade} onChange={e => { setGrade(e.target.value); setOpen(true); }}
          style={{
            padding: "0 6px", background: "#111", border: "1px solid #222",
            borderRadius: 8, color: "#666", fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace", cursor: "pointer", outline: "none",
          }}>
          <option value="all">All</option>
          {V_GRADES.slice(0, 12).map(g => <option key={g} value={g}>{g}</option>)}
        </select>
      </div>
      {open && (
        <div style={{
          position: "absolute", top: "calc(100% + 3px)", left: 0, right: 0,
          background: "#111", border: "1px solid #222", borderRadius: 8,
          maxHeight: 280, overflowY: "auto", zIndex: 100,
          boxShadow: "0 8px 32px rgba(0,0,0,0.7)",
        }}>
          {!results.length ? (
            <div style={{ padding: 14, textAlign: "center", fontSize: 11, color: "#444" }}>No climbs found</div>
          ) : results.map(c => (
            <button key={c.uuid}
              onClick={() => { onSelect(c); setOpen(false); setQ(c.name); }}
              style={{
                width: "100%", padding: "8px 10px", background: "transparent",
                border: "none", borderBottom: "1px solid #1a1a1a",
                cursor: "pointer", display: "flex", alignItems: "center", gap: 8,
                textAlign: "left", color: "#ddd",
              }}
              onMouseEnter={e => e.currentTarget.style.background = "#1a1a1a"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
              <span style={{
                fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
                fontWeight: 800, color: GRADE_COLORS[c.grade], minWidth: 26,
              }}>{c.grade}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</div>
                <div style={{ fontSize: 9, color: "#444", fontFamily: "'JetBrains Mono', monospace" }}>
                  {c.setter} · {c.angle}° · {c.ascents.toLocaleString()} sends
                </div>
              </div>
              <span style={{ fontSize: 9, color: "#333" }}>
                {"★".repeat(Math.round(c.quality))}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// --- Main ---
export default function App() {
  const [sel, setSel] = useState({});
  const [angle, setAngle] = useState(40);
  const [hover, setHover] = useState(null);
  const [tool, setTool] = useState("hand");
  const [climb, setClimb] = useState(null);
  const [mode, setMode] = useState("search");

  const clickHold = useCallback((id) => {
    if (mode !== "manual") return;
    setClimb(null);
    setSel(p => {
      const cur = p[id];
      if (cur === tool) { const n = { ...p }; delete n[id]; return n; }
      return { ...p, [id]: tool };
    });
  }, [tool, mode]);

  const selectClimb = useCallback((c) => {
    setClimb(c);
    setAngle(c.angle);
    setSel(parseLayout(c.layout));
  }, []);

  const clear = () => { setSel({}); setClimb(null); };

  const pred = useMemo(() => estimate(sel, angle), [sel, angle]);
  const verdict = useMemo(() => pred && climb ? getVerdict(pred.numeric, climb.grade) : null, [pred, climb]);
  const count = Object.values(sel).filter(r => r !== "none").length;

  return (
    <div style={{
      height: "100vh", background: "#050505", color: "#ddd",
      fontFamily: "'DM Sans', -apple-system, system-ui, sans-serif",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      {/* Top bar */}
      <div style={{
        padding: "8px 14px", borderBottom: "1px solid #151515",
        display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 24, height: 24, borderRadius: 5,
            background: "linear-gradient(135deg, #06d6e0, #e040a0)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 12,
          }}>🧗</div>
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: "-0.3px" }}>Boulder Grade Predictor</span>
        </div>
        <div style={{
          display: "flex", gap: 1, background: "#0a0a0a",
          borderRadius: 6, padding: 2, border: "1px solid #1a1a1a",
        }}>
          {[["search", "Search"], ["manual", "Manual"]].map(([k, l]) => (
            <button key={k} onClick={() => setMode(k)}
              style={{
                padding: "3px 10px", borderRadius: 4, border: "none",
                fontSize: 10, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
                cursor: "pointer",
                background: mode === k ? "#1a1a1a" : "transparent",
                color: mode === k ? "#ddd" : "#444",
              }}>{l}</button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Board panel */}
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", padding: 8,
          background: "#050505", position: "relative",
        }}>
          {/* Angle badge (like the real app) */}
          <div style={{
            position: "absolute", top: 12, left: 16,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 14, fontWeight: 700, color: "#06d6e0",
          }}>
            {angle}°
          </div>

          {mode === "manual" && (
            <div style={{
              display: "flex", gap: 3, marginBottom: 6,
              padding: 3, background: "#0a0a0a", borderRadius: 6, border: "1px solid #151515",
            }}>
              {ROLE_ORDER.map(r => (
                <button key={r} onClick={() => setTool(r)}
                  style={{
                    padding: "3px 10px", borderRadius: 4, border: "none",
                    fontSize: 10, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
                    cursor: "pointer",
                    background: tool === r ? ROLES[r].color : "transparent",
                    color: tool === r ? "#000" : "#444",
                  }}>{ROLES[r].label}</button>
              ))}
              <button onClick={clear} style={{
                padding: "3px 8px", borderRadius: 4, border: "1px solid #222",
                fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                cursor: "pointer", background: "transparent", color: "#444",
              }}>Clear</button>
            </div>
          )}

          {/* Kilter Board SVG */}
          <div style={{
            background: "#0a0a0a", borderRadius: 6,
            border: "1px solid #1a1a1a", padding: 2,
            boxShadow: "0 0 60px rgba(0,0,0,0.5)",
            maxHeight: "calc(100vh - 100px)",
            overflow: "hidden",
          }}>
            <svg width={BW} height={BH} viewBox={`0 0 ${BW} ${BH}`}
              style={{ display: "block" }}>
              <defs>
                {ROLE_ORDER.map(r => (
                  <filter key={r} id={`glow-${r}`}>
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                ))}
              </defs>
              <rect width={BW} height={BH} rx={4} fill="#0a0a0a"/>

              {/* All holds */}
              {HOLDS.map(h => {
                const role = sel[h.id];
                const active = role && role !== "none";
                const isHov = hover === h.id;
                return (
                  <g key={h.id}
                    style={{ cursor: mode === "manual" ? "pointer" : "default" }}
                    onClick={() => clickHold(h.id)}
                    onMouseEnter={() => setHover(h.id)}
                    onMouseLeave={() => setHover(null)}>
                    {active ? (
                      <ActiveHold x={h.x} y={h.y} role={role} seed={h.id} />
                    ) : (
                      <>
                        <HoldShape x={h.x} y={h.y} seed={h.id} />
                        {isHov && mode === "manual" && (
                          <ellipse cx={h.x} cy={h.y} rx={HOLD_R + 2} ry={HOLD_R + 2}
                            fill="none" stroke="#444" strokeWidth={1} opacity={0.5} />
                        )}
                      </>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Climb name below board */}
          {climb && (
            <div style={{
              marginTop: 6, textAlign: "center",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{climb.name}</div>
              <div style={{ fontSize: 10, color: "#555", fontFamily: "'JetBrains Mono', monospace" }}>
                {climb.setter} · {climb.grade} · ★{climb.quality}
              </div>
            </div>
          )}
          {!climb && count > 0 && (
            <div style={{ marginTop: 4, fontSize: 9, color: "#333", fontFamily: "'JetBrains Mono', monospace" }}>
              {count} holds selected
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div style={{
          width: 260, borderLeft: "1px solid #121212", padding: 10,
          display: "flex", flexDirection: "column", gap: 8,
          overflowY: "auto", flexShrink: 0,
          background: "#080808",
        }}>
          {mode === "search" && <Search onSelect={selectClimb} selected={climb} />}

          {/* Angle slider */}
          <div style={{
            background: "#0d0d0d", border: "1px solid #181818",
            borderRadius: 8, padding: "10px 12px",
          }}>
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6,
            }}>
              <span style={{
                fontSize: 9, color: "#444", fontFamily: "'JetBrains Mono', monospace",
                textTransform: "uppercase", letterSpacing: "0.5px",
              }}>Angle</span>
              <span style={{
                fontSize: 18, fontWeight: 800, fontFamily: "'JetBrains Mono', monospace",
                color: "#06d6e0",
              }}>{angle}°</span>
            </div>
            <input type="range" min={0} max={70} step={5} value={angle}
              onChange={e => setAngle(parseInt(e.target.value))}
              style={{ width: "100%", accentColor: "#06d6e0", height: 3 }} />
            <div style={{
              display: "flex", justifyContent: "space-between",
              fontSize: 8, color: "#2a2a2a", fontFamily: "'JetBrains Mono', monospace", marginTop: 2,
            }}>
              <span>0° vertical</span><span>70° roof</span>
            </div>
          </div>

          {/* Prediction */}
          {pred ? (
            <>
              {/* Grade display */}
              <div style={{
                background: "#0d0d0d", border: "1px solid #181818",
                borderRadius: 10, padding: "14px 16px",
              }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace", fontSize: 34, fontWeight: 800,
                    color: GRADE_COLORS[pred.grade],
                    textShadow: `0 0 24px ${GRADE_COLORS[pred.grade]}50`,
                    lineHeight: 1,
                  }}>{pred.grade}</span>
                  <span style={{ fontSize: 10, color: "#444", fontFamily: "'JetBrains Mono', monospace" }}>predicted</span>
                </div>

                {/* Confidence */}
                <div style={{ marginBottom: verdict ? 8 : 0 }}>
                  <div style={{
                    display: "flex", justifyContent: "space-between",
                    fontSize: 9, color: "#333", fontFamily: "'JetBrains Mono', monospace", marginBottom: 2,
                  }}>
                    <span>confidence</span><span>{Math.round(pred.confidence * 100)}%</span>
                  </div>
                  <div style={{ height: 3, background: "#151515", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{
                      height: "100%", width: `${pred.confidence * 100}%`,
                      background: GRADE_COLORS[pred.grade],
                      borderRadius: 2, transition: "width 0.3s",
                    }} />
                  </div>
                </div>

                {/* Verdict */}
                {verdict && (
                  <div style={{
                    display: "flex", alignItems: "center", gap: 6,
                    padding: "6px 8px", marginTop: 4,
                    background: `${verdict.color}0d`,
                    border: `1px solid ${verdict.color}25`,
                    borderRadius: 6,
                  }}>
                    <span style={{ fontSize: 14 }}>{verdict.emoji}</span>
                    <span style={{
                      fontSize: 10, color: verdict.color,
                      fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
                    }}>{verdict.text}</span>
                  </div>
                )}

                {/* Grade bar */}
                <div style={{ display: "flex", gap: 1.5, marginTop: 8 }}>
                  {V_GRADES.map(g => (
                    <div key={g} style={{
                      flex: 1, height: g === pred.grade ? 14 : 4,
                      background: g === pred.grade ? GRADE_COLORS[g] : "#151515",
                      borderRadius: 1.5, transition: "all 0.3s",
                      opacity: g === pred.grade ? 1 : 0.25,
                    }} title={g} />
                  ))}
                </div>
              </div>

              {/* Features */}
              <div style={{
                background: "#0d0d0d", border: "1px solid #181818",
                borderRadius: 8, padding: "10px 12px",
              }}>
                <div style={{
                  fontSize: 8, color: "#333", fontFamily: "'JetBrains Mono', monospace",
                  textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6,
                }}>Extracted Features</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                  {[
                    ["Holds", pred.features.holds, ""],
                    ["Hands", pred.features.hands, ""],
                    ["Feet", pred.features.feet, ""],
                    ["Avg move", pred.features.avgMove, ""],
                    ["Max move", pred.features.maxMove, ""],
                    ["Angle", pred.features.angle, "°"],
                  ].map(([l, v, u]) => (
                    <div key={l} style={{
                      background: "#080808", borderRadius: 4, padding: "4px 6px",
                      border: "1px solid #131313",
                    }}>
                      <div style={{
                        fontSize: 7, color: "#333", fontFamily: "'JetBrains Mono', monospace",
                        textTransform: "uppercase",
                      }}>{l}</div>
                      <div style={{
                        fontSize: 14, fontWeight: 700, color: "#888",
                        fontFamily: "'JetBrains Mono', monospace",
                      }}>{v}<span style={{ fontSize: 8, color: "#333" }}>{u}</span></div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div style={{
              background: "#0d0d0d", border: "1px dashed #181818",
              borderRadius: 8, padding: 20, textAlign: "center",
            }}>
              <div style={{ fontSize: 20, marginBottom: 4, opacity: 0.5 }}>🪨</div>
              <p style={{ fontSize: 11, color: "#333", margin: 0, lineHeight: 1.4 }}>
                {mode === "search" ? "Search for a climb above" : "Place at least 3 holds"}
              </p>
            </div>
          )}

          {/* Legend */}
          <div style={{ borderTop: "1px solid #131313", paddingTop: 8 }}>
            <div style={{
              fontSize: 7, color: "#2a2a2a", fontFamily: "'JetBrains Mono', monospace",
              textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4,
            }}>Hold roles</div>
            <div style={{ display: "flex", gap: 10 }}>
              {ROLE_ORDER.map(r => (
                <div key={r} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                  <div style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: ROLES[r].color,
                    boxShadow: `0 0 4px ${ROLES[r].glow}80`,
                  }} />
                  <span style={{ fontSize: 8, color: "#444", fontFamily: "'JetBrains Mono', monospace" }}>
                    {ROLES[r].label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            fontSize: 8, color: "#1a1a1a", fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 1.4, borderTop: "1px solid #131313", paddingTop: 6,
          }}>
            Demo uses heuristic estimation. Production: XGBoost trained on 100k+ ascents.
          </div>
        </div>
      </div>
    </div>
  );
}

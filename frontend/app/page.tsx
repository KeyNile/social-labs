"use client";
import { useState } from "react";
import PersonaCard from "@/components/PersonaCard";
import TrajectoryChart from "@/components/TrajectoryChart";
import SimulationPanel from "@/components/SimulationPanel";

type Mode = "user" | "researcher";

interface PersonaSummary {
  id: string;
  match_score: number;
  summary: string;
  key_events: string[];
}

interface TimelineEntry {
  year: number;
  income?: number;
  education?: string;
  events: string[];
}

interface AgentResponse {
  id: string;
  response: string;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("user");
  const [context, setContext] = useState("");
  const [question, setQuestion] = useState("");
  const [personas, setPersonas] = useState<PersonaSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [timelines, setTimelines] = useState<{ personaId: string; data: TimelineEntry[] }[]>([]);
  const [advice, setAdvice] = useState<AgentResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "dev-key";
  const headers = { "Content-Type": "application/json", "X-API-Key": API_KEY };

  async function handleQuery() {
    if (!context.trim()) return;
    setLoading(true);
    setError("");
    setPersonas([]);
    setSelectedIds([]);
    setTimelines([]);
    setAdvice([]);
    try {
      const res = await fetch(`${API_BASE}/v1/query`, {
        method: "POST",
        headers,
        body: JSON.stringify({ context, n: 5 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPersonas(data.personas);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleGetAdvice() {
    if (selectedIds.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const [trajResults, simResult] = await Promise.all([
        Promise.all(
          selectedIds.map(id =>
            fetch(`${API_BASE}/v1/personas/${id}/trajectory`, { headers })
              .then(r => r.json())
              .then(d => ({ personaId: id, data: d.timeline as TimelineEntry[] }))
          )
        ),
        fetch(`${API_BASE}/v1/simulate`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            persona_ids: selectedIds,
            question: question || context,
            mode: "individual",
          }),
        }).then(r => r.json()),
      ]);
      setTimelines(trajResults);
      setAdvice(simResult.responses);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function toggleSelect(id: string) {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id].slice(0, 5)
    );
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-100 px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-bold text-gray-900">Social Labs</h1>
        <div className="flex gap-1 bg-gray-100 rounded-full p-1">
          {(["user", "researcher"] as Mode[]).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 rounded-full text-sm transition-colors ${
                mode === m ? "bg-white shadow text-gray-900" : "text-gray-500"
              }`}
            >
              {m === "user" ? "일반" : "연구자"}
            </button>
          ))}
        </div>
      </header>

      <div className="p-6">
        {mode === "user" ? (
          <div className="max-w-5xl mx-auto space-y-6">
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                상황을 설명해 주세요
              </label>
              <textarea
                value={context}
                onChange={e => setContext(e.target.value)}
                rows={3}
                className="w-full px-4 py-3 border border-gray-200 rounded-lg text-sm resize-none"
                placeholder="저는 38세 마케터입니다. 15년간 일했는데 창업을 고민하고 있어요..."
              />
              <input
                value={question}
                onChange={e => setQuestion(e.target.value)}
                className="w-full mt-2 px-4 py-2 border border-gray-200 rounded-lg text-sm"
                placeholder="구체적인 질문 (선택) — 비워두면 상황 설명이 질문으로 사용됩니다"
              />
              <button
                onClick={handleQuery}
                disabled={loading}
                className="mt-3 px-6 py-2.5 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50"
              >
                {loading ? "검색 중..." : "유사 경험자 찾기 →"}
              </button>
              {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
            </div>

            {personas.length > 0 && (
              <div className="grid grid-cols-3 gap-6">
                <div className="space-y-3">
                  <p className="text-xs text-gray-500 font-medium">
                    유사 페르소나 — 최대 5명 선택
                  </p>
                  {personas.map(p => (
                    <PersonaCard
                      key={p.id}
                      persona={p}
                      selected={selectedIds.includes(p.id)}
                      onClick={() => toggleSelect(p.id)}
                    />
                  ))}
                  {selectedIds.length > 0 && (
                    <button
                      onClick={handleGetAdvice}
                      disabled={loading}
                      className="w-full py-2.5 bg-gray-900 text-white rounded-lg text-sm disabled:opacity-50"
                    >
                      {loading ? "조언 생성 중..." : `${selectedIds.length}명에게 조언 듣기`}
                    </button>
                  )}
                </div>

                <div className="col-span-2 space-y-4">
                  {timelines.length > 0 && (
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
                      <TrajectoryChart timelines={timelines} />
                    </div>
                  )}
                  {advice.length > 0 && (
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-4">
                      <h2 className="font-semibold text-gray-800">에이전트 조언</h2>
                      {advice.map(r => (
                        <div key={r.id} className="pl-4 border-l-2 border-blue-200">
                          <span className="font-mono text-xs text-gray-400">{r.id}</span>
                          <p className="text-sm text-gray-800 mt-1 leading-relaxed">{r.response}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          <SimulationPanel apiBase={API_BASE} headers={headers} />
        )}
      </div>
    </main>
  );
}

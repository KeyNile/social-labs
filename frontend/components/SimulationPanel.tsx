"use client";
import { useState } from "react";

interface AgentResponse {
  id: string;
  response: string;
  round: number;
}

interface Props {
  apiBase: string;
  headers: Record<string, string>;
}

export default function SimulationPanel({ apiBase, headers }: Props) {
  const [personaInput, setPersonaInput] = useState("ID_001, ID_002, ID_003");
  const [question, setQuestion] = useState("");
  const [rounds, setRounds] = useState(3);
  const [allRounds, setAllRounds] = useState<AgentResponse[][]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleStart() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    const ids = personaInput.split(",").map(s => s.trim()).filter(Boolean);
    try {
      const res = await fetch(`${apiBase}/v1/simulate`, {
        method: "POST",
        headers,
        body: JSON.stringify({ persona_ids: ids, question, mode: "discussion", rounds }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAllRounds(data.rounds ?? []);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleExportJSON() {
    const blob = new Blob([JSON.stringify(allRounds, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "simulation.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleExportCSV() {
    const rows = [["round", "persona_id", "response"]];
    allRounds.forEach((round, ri) =>
      round.forEach(r =>
        rows.push([String(ri + 1), r.id, `"${r.response.replace(/"/g, '""')}"`])
      )
    );
    const blob = new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "simulation.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="font-semibold mb-4 text-gray-800">시나리오 설정</h2>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-gray-500">페르소나 ID (쉼표 구분)</label>
            <input
              value={personaInput}
              onChange={e => setPersonaInput(e.target.value)}
              className="w-full mt-1 px-3 py-2 border border-gray-200 rounded-lg text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">질문 / 시나리오</label>
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={3}
              className="w-full mt-1 px-3 py-2 border border-gray-200 rounded-lg text-sm resize-none"
              placeholder="What was the hardest decision in your career?"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">라운드 수: {rounds}</label>
            <input
              type="range" min={1} max={10} value={rounds}
              onChange={e => setRounds(Number(e.target.value))}
              className="w-full mt-1 accent-blue-600"
            />
          </div>
        </div>
        <div className="flex gap-3 mt-5">
          <button
            onClick={handleStart}
            disabled={loading}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50"
          >
            {loading ? "시뮬레이션 중..." : "시작"}
          </button>
          {allRounds.length > 0 && (
            <>
              <button onClick={handleExportJSON}
                className="px-4 py-2 border border-gray-200 rounded-lg text-sm">
                JSON
              </button>
              <button onClick={handleExportCSV}
                className="px-4 py-2 border border-gray-200 rounded-lg text-sm">
                CSV
              </button>
            </>
          )}
        </div>
        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>

      {allRounds.map((round, ri) => (
        <div key={ri} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-500 mb-3">Round {ri + 1}</h3>
          <div className="space-y-4">
            {round.map(resp => (
              <div key={resp.id} className="pl-4 border-l-2 border-gray-200">
                <span className="font-mono text-xs text-gray-400">{resp.id}</span>
                <p className="text-sm text-gray-800 mt-1 leading-relaxed">{resp.response}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

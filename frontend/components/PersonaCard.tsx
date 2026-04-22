interface PersonaSummary {
  id: string;
  match_score: number;
  summary: string;
  key_events: string[];
}

interface Props {
  persona: PersonaSummary;
  selected: boolean;
  onClick: () => void;
}

export default function PersonaCard({ persona, selected, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
        selected
          ? "border-blue-500 bg-blue-50"
          : "border-gray-200 hover:border-gray-300 bg-white"
      }`}
    >
      <div className="flex justify-between items-center mb-1">
        <span className="font-mono text-xs text-gray-400">{persona.id}</span>
        <span className="text-sm font-semibold text-blue-600">
          {(persona.match_score * 100).toFixed(0)}% 매칭
        </span>
      </div>
      <p className="text-sm text-gray-700">{persona.summary}</p>
      {persona.key_events.length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {persona.key_events.map((e, i) => (
            <li key={i} className="text-xs text-gray-500">▸ {e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

interface TimelineEntry {
  year: number;
  income?: number;
  education?: string;
  events: string[];
}

interface PersonaTimeline {
  personaId: string;
  data: TimelineEntry[];
}

interface Props {
  timelines: PersonaTimeline[];
}

const COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"];

export default function TrajectoryChart({ timelines }: Props) {
  const allYears = [...new Set(timelines.flatMap(t => t.data.map(d => d.year)))].sort(
    (a, b) => a - b
  );

  const chartData = allYears.map(year => {
    const entry: Record<string, number | string> = { year: String(year) };
    timelines.forEach(({ personaId, data }) => {
      const point = data.find(d => d.year === year);
      if (point?.income != null) {
        entry[personaId] = point.income;
      }
    });
    return entry;
  });

  return (
    <div className="w-full">
      <p className="text-xs text-gray-500 mb-2">소득 궤적 비교 (연도별)</p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="year" tick={{ fontSize: 11 }} />
          <YAxis
            tickFormatter={v => `$${(Number(v) / 1000).toFixed(0)}k`}
            tick={{ fontSize: 11 }}
          />
          <Tooltip formatter={(v) => [`$${Number(v).toLocaleString()}`, ""]} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {timelines.map(({ personaId }, i) => (
            <Line
              key={personaId}
              type="monotone"
              dataKey={personaId}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              connectNulls
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

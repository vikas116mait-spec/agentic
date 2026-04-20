"use client";

type LossPoint = { step: number; loss: number };

type LossChartProps = {
  data: LossPoint[];
  height?: number;
  className?: string;
};

export function LossChart({ data, height = 80, className = "" }: LossChartProps) {
  if (!data || data.length < 2) {
    return (
      <div
        className={`flex items-center justify-center text-xs text-black/35 ${className}`}
        style={{ height }}
      >
        Waiting for training data...
      </div>
    );
  }

  const width = 400;
  const padX = 4;
  const padY = 6;

  const minLoss = Math.min(...data.map((d) => d.loss));
  const maxLoss = Math.max(...data.map((d) => d.loss));
  const lossRange = maxLoss - minLoss || 1;
  const minStep = data[0].step;
  const maxStep = data[data.length - 1].step;
  const stepRange = maxStep - minStep || 1;

  const toX = (step: number) =>
    padX + ((step - minStep) / stepRange) * (width - padX * 2);
  const toY = (loss: number) =>
    padY + (1 - (loss - minLoss) / lossRange) * (height - padY * 2);

  const points = data.map((d) => `${toX(d.step)},${toY(d.loss)}`).join(" ");
  const lastPoint = data[data.length - 1];
  const dotX = toX(lastPoint.step);
  const dotY = toY(lastPoint.loss);

  // area fill path: line points + close down to baseline
  const areaPath = [
    `M ${toX(data[0].step)},${toY(data[0].loss)}`,
    ...data.slice(1).map((d) => `L ${toX(d.step)},${toY(d.loss)}`),
    `L ${toX(lastPoint.step)},${height - padY}`,
    `L ${toX(data[0].step)},${height - padY}`,
    "Z",
  ].join(" ");

  return (
    <div className={`w-full overflow-hidden ${className}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="w-full"
        style={{ height }}
      >
        <defs>
          <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-brand, #16a34a)" stopOpacity="0.18" />
            <stop offset="100%" stopColor="var(--color-brand, #16a34a)" stopOpacity="0.01" />
          </linearGradient>
        </defs>

        {/* Area fill */}
        <path d={areaPath} fill="url(#loss-fill)" />

        {/* Line */}
        <polyline
          points={points}
          fill="none"
          stroke="var(--color-brand, #16a34a)"
          strokeWidth="1.8"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Current point dot */}
        <circle cx={dotX} cy={dotY} r="3.5" fill="var(--color-brand, #16a34a)" />
        <circle cx={dotX} cy={dotY} r="6" fill="var(--color-brand, #16a34a)" fillOpacity="0.18" />
      </svg>

      <div className="mt-1 flex items-center justify-between px-1 text-xs text-black/40">
        <span>step {data[0].step}</span>
        <span>loss {lastPoint.loss.toFixed(4)}</span>
        <span>step {lastPoint.step}</span>
      </div>
    </div>
  );
}

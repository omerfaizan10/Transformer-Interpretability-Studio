function intensityStyle(value) {
  const safeValue = Number.isFinite(value) ? value : 0
  const alpha = Math.max(0.05, Math.min(0.98, safeValue * 3.0))

  return {
    background: `linear-gradient(135deg, rgba(14,165,233,${alpha}), rgba(16,185,129,${alpha * 0.82}))`,
  }
}

function shorten(token, max = 10) {
  if (!token) return ''
  return token.length > max ? `${token.slice(0, max - 1)}…` : token
}

function cellSizeFor(size) {
  if (size <= 4) return 46
  if (size <= 8) return 38
  if (size <= 14) return 30
  if (size <= 24) return 24
  return 18
}

export default function Heatmap({ matrix = [], tokens = [], title = 'Attention Heatmap' }) {
  const size = matrix.length
  const cellSize = cellSizeFor(size)
  const labelWidth = size <= 8 ? 76 : 88
  const gridTemplate = `${labelWidth}px repeat(${size}, ${cellSize}px)`

  if (!matrix.length || !tokens.length) {
    return (
      <div className="studio-card rounded-3xl p-6 text-slate-400">
        No heatmap data yet.
      </div>
    )
  }

  return (
    <div className="studio-card rounded-3xl p-5">
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="section-kicker">Attention matrix</p>
          <h3 className="text-xl font-semibold tracking-tight text-white">{title}</h3>
          <p className="mt-2 text-sm text-slate-500">
            Rows are query tokens. Columns are key tokens. Brighter cells mean stronger attention.
          </p>
        </div>
        <div className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs text-slate-400">
          {size} × {size}
        </div>
      </div>

      <div className="max-h-[560px] overflow-auto rounded-2xl border border-slate-800 bg-slate-950 p-3">
        <div
          className="grid gap-1 text-[10px]"
          style={{ gridTemplateColumns: gridTemplate }}
        >
          <div />
          {tokens.map((token, index) => (
            <div
              key={`col-${index}`}
              className="sticky top-0 z-10 flex items-center justify-center rounded-lg bg-slate-900 px-1 text-center text-slate-400"
              style={{ width: cellSize, height: 42 }}
              title={`${index}: ${token}`}
            >
              <span className="-rotate-45 whitespace-nowrap text-[9px]">{index}:{shorten(token, 8)}</span>
            </div>
          ))}

          {matrix.map((row, rowIndex) => (
            <div key={`rowwrap-${rowIndex}`} className="contents">
              <div
                className="sticky left-0 z-10 flex items-center rounded-lg bg-slate-900 px-2 text-slate-400"
                style={{ width: labelWidth, height: cellSize }}
                title={`${rowIndex}: ${tokens[rowIndex]}`}
              >
                <span className="truncate">{rowIndex}:{shorten(tokens[rowIndex], 9)}</span>
              </div>

              {row.map((value, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="heat-cell rounded-md border border-slate-800 transition duration-150 hover:scale-125 hover:border-slate-200 hover:z-20"
                  style={{ ...intensityStyle(value), width: cellSize, height: cellSize }}
                  title={`query ${rowIndex}:${tokens[rowIndex]} → key ${colIndex}:${tokens[colIndex]} | weight ${Number(value).toFixed(4)}`}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3 text-xs text-slate-500">
        <span>low</span>
        <div className="h-2 w-44 rounded-full bg-gradient-to-r from-slate-800 via-sky-500 to-emerald-400" />
        <span>high attention</span>
      </div>
    </div>
  )
}

export default function TokenRibbon({ tokens = [] }) {
  if (!tokens.length) {
    return <p className="text-sm text-slate-500">No tokens yet.</p>
  }

  return (
    <div className="flex flex-wrap gap-2">
      {tokens.map((token, index) => (
        <span
          key={`${token}-${index}`}
          className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-200"
        >
          <span className="text-slate-500">{index}</span>
          <span className="mx-1 text-slate-600">/</span>
          <span>{token}</span>
        </span>
      ))}
    </div>
  )
}

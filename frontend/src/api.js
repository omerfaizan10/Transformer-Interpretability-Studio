const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    ...options,
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `Request failed: ${response.status}`)
  }

  return response.json()
}

export function getConfig() {
  return request('/api/config')
}

export function getHealth() {
  return request('/api/health')
}

export function analyzeAttention(payload) {
  return request('/api/analyze', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function generateText(payload) {
  return request('/api/generate', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getPositionalEncoding(positions = 48, dims = 48) {
  return request(`/api/positional-encoding?positions=${positions}&dims=${dims}`)
}

export function explainAttention(payload) {
  return request('/api/explain-attention', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function testClaude() {
  return request('/api/claude-test')
}

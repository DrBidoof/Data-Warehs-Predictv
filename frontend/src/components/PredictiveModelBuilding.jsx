import React, { useState, useEffect, useCallback } from 'react'

export default function PredictiveModelBuilding() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [index, setIndex] = useState(0)

  async function run() {
    setLoading(true)
    setResult(null)
    setIndex(0)
    try {
      const r = await fetch('http://localhost:5000/run/predictive')
      const j = await r.json()
      setResult(j)
    } catch (err) {
      setResult({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  const images = (result && result.images) || []
  const prev = useCallback(() => setIndex((i) => Math.max(0, i - 1)), [])
  const next = useCallback(() => setIndex((i) => Math.min(images.length - 1, i + 1)), [images.length])

  useEffect(() => setIndex(0), [result && result.images && result.images.length])

  return (
    <div>
      <h3>Predictive model building</h3>
      <p>Runs `Predictive model building.py` and returns stdout/stderr and figures.</p>
      <button onClick={run} disabled={loading}>{loading ? 'Running...' : 'Run'}</button>
      {result && (
        <div style={{ marginTop: 12 }}>
          <strong>Exit code:</strong> {result.code ?? 'n/a'}
          <div style={{ marginTop: 8 }}>
            <strong>Stdout</strong>
            <pre>{result.stdout ?? ''}</pre>
            <strong>Stderr</strong>
            <pre>{result.stderr ?? ''}</pre>
            {result.error && (<div style={{ color: 'red' }}>Error: {result.error}</div>)}

            {images.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button onClick={prev} disabled={index <= 0}>Prev</button>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ marginBottom: 6 }}><em>{index + 1} / {images.length}</em></div>
                    <img src={encodeURI(`http://localhost:5000${images[index]}`)} alt={images[index]} style={{ maxWidth: 640, maxHeight: 480, border: '1px solid #ddd' }} />
                  </div>
                  <button onClick={next} disabled={index >= images.length - 1}>Next</button>
                </div>

                <div style={{ marginTop: 10, display: 'flex', gap: 8, overflowX: 'auto', paddingTop: 8 }}>
                  {images.map((src, i) => (
                    <img key={src} src={encodeURI(`http://localhost:5000${src}`)} alt={src} style={{ height: 60, cursor: 'pointer', border: i === index ? '2px solid #007acc' : '1px solid #ddd' }} onClick={() => setIndex(i)} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

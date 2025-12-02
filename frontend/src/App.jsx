import React, { useState } from 'react'
import DataExploration from './components/DataExploration'
import DataModelling from './components/DataModelling'
import PredictiveModelBuilding from './components/PredictiveModelBuilding'

export default function App() {
  const [tab, setTab] = useState('exploration')

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: 20 }}>
      <h2>Data-Warehs-Predictv — Interface</h2>
      <div style={{ marginBottom: 12 }}>
        <button onClick={() => setTab('exploration')} style={{ marginRight: 8 }}>Data exploration</button>
        <button onClick={() => setTab('modelling')} style={{ marginRight: 8 }}>Data modelling</button>
        <button onClick={() => setTab('predictive')}>Predictive model building</button>
      </div>

      <div style={{ border: '1px solid #ddd', padding: 16, borderRadius: 6 }}>
        {tab === 'exploration' && <DataExploration />}
        {tab === 'modelling' && <DataModelling />}
        {tab === 'predictive' && <PredictiveModelBuilding />}
      </div>
    </div>
  )
}

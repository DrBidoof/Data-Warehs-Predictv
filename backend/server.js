const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const repoRoot = path.resolve(__dirname, '..');

const scripts = {
  exploration: path.join(repoRoot, 'Data exploration.py'),
  modelling: path.join(repoRoot, 'Data modelling.py'),
  predictive: path.join(repoRoot, 'Predictive model building.py')
};

function runScript(scriptPath, res) {
  const pythonCmd = 'python';
  const proc = spawn(pythonCmd, [scriptPath], { cwd: repoRoot });

  let stdout = '';
  let stderr = '';

  proc.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  proc.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  proc.on('close', (code) => {
    res.json({ code, stdout, stderr });
  });

  proc.on('error', (err) => {
    res.status(500).json({ error: err.message });
  });
}

app.get('/run/:script', (req, res) => {
  const key = req.params.script;
  const scriptPath = scripts[key];
  if (!scriptPath) return res.status(404).json({ error: 'Unknown script' });
  runScript(scriptPath, res);
});

const port = process.env.PORT || 5000;
app.listen(port, () => {
  console.log(`Backend running on port ${port}`);
});

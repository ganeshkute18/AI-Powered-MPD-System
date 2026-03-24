const streamImg = document.getElementById('stream');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const sourceInput = document.getElementById('sourceInput');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const fpsBadge = document.getElementById('fpsBadge');
const logsList = document.getElementById('logsList');
const registerForm = document.getElementById('registerForm');

function setStatus(running) {
  statusText.textContent = running ? 'Running' : 'Stopped';
  statusDot.classList.toggle('running', running);
  statusDot.classList.toggle('stopped', !running);
}

async function startStream() {
  const sourceRaw = sourceInput.value.trim();
  const source = /^\d+$/.test(sourceRaw) ? Number(sourceRaw) : sourceRaw;
  const res = await fetch('/api/stream/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source })
  });
  const data = await res.json();
  if (!data.success) {
    alert(data.message || 'Could not start stream');
    return;
  }
  streamImg.src = `/video_feed?t=${Date.now()}`;
  setStatus(true);
}

async function stopStream() {
  const res = await fetch('/api/stream/stop', { method: 'POST' });
  const data = await res.json();
  if (!data.success) {
    alert(data.message || 'Could not stop stream');
    return;
  }
  streamImg.src = '';
  setStatus(false);
}

async function refreshLogs() {
  try {
    const res = await fetch('/api/logs');
    const data = await res.json();
    fpsBadge.textContent = `FPS: ${Number(data.fps || 0).toFixed(1)}`;
    setStatus(data.status === 'running');

    logsList.innerHTML = '';
    data.logs.forEach((item) => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${item.name}</strong> (${item.score.toFixed(2)})<br/><small>${item.timestamp}</small>`;
      logsList.appendChild(li);
    });
  } catch (err) {
    console.error(err);
  }
}

registerForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(registerForm);
  const res = await fetch('/register', { method: 'POST', body: formData });
  const data = await res.json();
  if (!data.success) {
    alert(data.message || 'Registration failed');
    return;
  }
  alert(data.message);
  registerForm.reset();
});

startBtn.addEventListener('click', startStream);
stopBtn.addEventListener('click', stopStream);

setInterval(refreshLogs, 1500);
refreshLogs();

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed.");
  }
  return response;
}

export async function predictVideo(file) {
  const formData = new FormData();
  formData.append("video", file);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response).then((res) => res.json());
}

export async function compareVideos(fileA, fileB) {
  const formData = new FormData();
  formData.append("video_a", fileA);
  formData.append("video_b", fileB);

  const response = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response).then((res) => res.json());
}

export async function downloadReport(payload) {
  const response = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const verified = await parseResponse(response);
  return verified.blob();
}

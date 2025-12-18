import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

export async function recommend(query) {
  const res = await axios.post(`${API_BASE}/recommend`, {
    query: query,
    top_k: 10,
  });
  return res.data.recommended_assessments;
}

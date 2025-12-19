import axios from "axios";

const API_BASE = "https://shl-backend-wvnd.onrender.com";

export async function recommend(query) {
  const res = await axios.post(`${API_BASE}/recommend`, {
    query: query,
    top_k: 10,
  });
  return res.data.recommended_assessments;
}
import axios from "axios";

const API_BASE = "https://shl-backend-wvnd.onrender.com";

function extractTopK(query) {
  const match = query.match(/\b([5-9]|10)\b/); 
  return match ? parseInt(match[1], 10) : 10;
}

function cleanQuery(query) {
  return query.replace(/\b([5-9]|10)\b/g, "").trim();
}

export async function recommend(query) {
  const topK = extractTopK(query);
  const cleanedQuery = cleanQuery(query);

  const res = await axios.post(`${API_BASE}/recommend`, {
    query: cleanedQuery,
    top_k: topK,
  });

  return res.data.recommended_assessments;
}

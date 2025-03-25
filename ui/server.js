require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');

const app = express();
const PORT = process.env.PORT || 5000;
const API_KEY = process.env.GEMINI_API_KEY;

app.use(express.json());
app.use(cors());

app.post('/get-response', async (req, res) => {
    const { query, type } = req.body;

    const url = `https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=${API_KEY}`;

    const requestBody = {
        contents: [{ role: "user", parts: [{ text: query }] }]
    };

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();
        const textResponse = data.candidates?.[0]?.content?.parts?.[0]?.text || "No response.";

        if (type === "image") {
            return res.json({ imageUrl: `https://source.unsplash.com/featured/?${encodeURIComponent(query)}` });
        }

        res.json({ text: textResponse });

    } catch (error) {
        console.error("Error fetching AI response:", error);
        res.status(500).json({ error: "Failed to fetch AI response." });
    }
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

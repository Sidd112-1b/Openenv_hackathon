import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API routes for the OpenEnv simulation
  // Since we can't run Python easily in this Node environment, 
  // we'll implement a TypeScript version of the environment for the preview.
  
  let currentState: any = null;

  app.post("/api/reset", (req, res) => {
    const { taskId } = req.body;
    // Simple mock reset
    currentState = {
      taskId: taskId || "easy-password-reset",
      step: 0,
      content: taskId === "easy-password-reset" ? "Hi, I forgot my password." : "I have a billing issue.",
      history: [],
      done: false
    };
    res.json(currentState);
  });

  app.post("/api/step", (req, res) => {
    const { action } = req.body;
    if (!currentState) return res.status(400).json({ error: "Call reset first" });

    currentState.step += 1;
    currentState.history.push(`Agent: ${action.tool} - ${JSON.stringify(action.args)}`);
    
    let reward = 0;
    if (action.tool === "reply" && currentState.taskId === "easy-password-reset") {
      reward = 1.0;
      currentState.done = true;
    } else if (action.tool === "search") {
      reward = 0.2;
    }

    res.json({
      observation: currentState,
      reward: { value: reward, done: currentState.done },
      done: currentState.done
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();

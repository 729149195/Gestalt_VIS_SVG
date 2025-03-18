import express from "express";
import cors from "cors";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import fs from "fs/promises";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;

let submissionCount = 0;

// 简化 CORS 配置，允许所有来源
app.use(cors());

app.use(express.json());

// 确保目录存在
const questionnaireDir = join(__dirname, "../public/Test_QuestionnaireData");

// 添加错误处理中间件
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Accept");
  next();
});

// 静态文件服务
app.use("/questionnaire", express.static(questionnaireDir));

// 获取问卷列表
app.get("/questionnaire/list", async (req, res) => {
  try {
    // 检查目录是否存在
    try {
      await fs.access(questionnaireDir);
    } catch (error) {
      console.error("Directory does not exist:", questionnaireDir);
      return res
        .status(500)
        .json({ error: "Questionnaire directory not found" });
    }

    const files = await fs.readdir(questionnaireDir);
    console.log("Found files:", files); // 调试日志

    const questionnaireIds = files
      .filter((file) => file.endsWith(".json"))
      .map((file) => file.replace(".json", ""));
    console.log("Questionnaire IDs:", questionnaireIds); // 调试日志

    res.json(questionnaireIds);
  } catch (error) {
    console.error("Error reading questionnaire directory:", error);
    res.status(500).json({ error: "Failed to load questionnaire list" });
  }
});

// API 路由
app.get("/api/count", (req, res) => {
  res.json({ count: submissionCount });
});

app.post("/api/increment", (req, res) => {
  submissionCount++;
  res.json({ count: submissionCount });
});

app.post("/api/reset", (req, res) => {
  submissionCount = 0;
  res.json({ count: submissionCount });
});

// 错误处理中间件
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Something broke!" });
});

// 启动服务器
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  console.log(`Serving questionnaire data from: ${questionnaireDir}`);
  // 启动时检查目录
  fs.access(questionnaireDir)
    .then(() => console.log(` ${questionnaireDir}  directory exists`))
    .catch(() =>
      console.error("WARNING: QuestionnaireData2 directory not found!")
    );
});

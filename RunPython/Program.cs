using System;
using System.Diagnostics;
using System.IO;

namespace RunPythonScript
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. 配置Python解释器路径和脚本路径
            string pythonPath = "python3"; // macOS/Linux使用python3，Windows使用python
            string scriptPath = "/Users/hujunyan/TERL/TERL_Model/visualization_mainwindow/Apply_data.py";

            // // 2. 设置默认参数
            // int pursuerNum = 30;
            // int evaderNum = 6;
            // int perception = 20;
            // int obstacleNum = 5;

            // // 3. 如果有命令行参数，则覆盖默认值
            // if (args.Length >= 4)
            // {
            //     int.TryParse(args[0], out pursuerNum);
            //     int.TryParse(args[1], out evaderNum);
            //     int.TryParse(args[2], out perception);
            //     int.TryParse(args[3], out obstacleNum);
            // }

            // // 2. 创建ProcessStartInfo对象，配置进程启动信息
            // string arguments = $"\"{scriptPath}\" --pursuer-num {pursuerNum} --evader-num {evaderNum} --perception {perception} --obstacle-num {obstacleNum}";
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = $"\"{scriptPath}\"", // 脚本路径用引号包裹，避免空格问题
                UseShellExecute = true, // 使用shell执行（适合GUI应用）
                RedirectStandardOutput = false, // GUI应用不需要重定向输出
                RedirectStandardError = false,
                CreateNoWindow = false, // 显示Python脚本的GUI窗口
                WorkingDirectory = Path.GetDirectoryName(scriptPath) // 设置工作目录为脚本所在目录
            };

            try
            {
                // 3. 启动进程执行Python脚本
                using (Process process = new Process { StartInfo = startInfo })
                {
                    Console.WriteLine("正在启动Python GUI应用...");
                    process.Start();
                    
                    // 可选：等待进程结束（如果需要同步执行）
                    process.WaitForExit();
                    // Console.WriteLine($"进程已结束，退出码: {process.ExitCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"执行脚本失败: {ex.Message}");
                Console.WriteLine($"请检查Python路径和脚本路径是否正确，以及依赖是否已安装。");
            }
        }
    }
}
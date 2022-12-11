package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.enumerates.LogType;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class LogUtil {
    private static LogType logType = LogType.DEBUG;
    private static boolean fileOut = true;

    private static Map<String, BufferedWriter> writerMap = new HashMap<>();
    private static Map<String, Boolean> withStdOutMap = new HashMap<>();

    public static boolean isFileOut() {
        return fileOut;
    }

    public static void setFileOut(boolean fileOut) {
        LogUtil.fileOut = fileOut;
    }

    public static void addWriter(String key, String filename, boolean withStdOut) {
        if (!QAArgs.isWritingLogs())
            return;
        try {
            writerMap.put(key, new BufferedWriter(new FileWriter(filename)));
            withStdOutMap.put(key, withStdOut);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void printInfo(String writer, String str) {
        print(writer, str, LogType.INFO);
    }

    public static void printlnInfo(String writer, String str) {
        println(writer, str, LogType.INFO);
    }

    public static void printDebug(String writer, String str) {
        print(writer, str, LogType.DEBUG);
    }

    public static void printlnDebug(String writer, String str) {
        println(writer, str, LogType.DEBUG);
    }

    public static void printError(String writer, String str) {
        print(writer, str, LogType.ERROR);
    }

    public static void printlnError(String writer, String str) {
        println(writer, str, LogType.ERROR);
    }

    public static void print(String writer, String str) {
        print(writer, str, LogType.None);
    }

    public static void print(String writer, String str, LogType logType) {
        if (!QAArgs.isWritingLogs())
            return;
        if (logType == LogType.INFO) {
            str = "[INFO] " + str;
        } else if (logType == LogType.DEBUG) {
            str = "[DEBUG] " + str;
        } else if (logType == LogType.ERROR) {
            str = "[ERROR] " + str;
        } else if (logType == LogType.WARN) {
            str = "[WARN] " + str;
        }

        if (withStdOutMap.containsKey(writer) && withStdOutMap.get(writer))
            System.out.print(str);

        if (!fileOut || !writerMap.containsKey(writer)) return;
        try {
            writerMap.get(writer).write(str);
            writerMap.get(writer).flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void println(String writer, String str) {
        print(writer, str + "\n", LogType.None);
    }

    public static void println(String writer, String str, LogType logType) {
        print(writer, str + "\n", logType);
    }

    public static void closeWriter(String writer) {
        if (!QAArgs.isWritingLogs())
            return;
        if (writerMap.containsKey(writer)) {
            try {
                writerMap.get(writer).close();
                writerMap.remove(writer);
                withStdOutMap.remove(writer);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void closeAllWriters() {
        if (!QAArgs.isWritingLogs())
            return;
        for (Map.Entry<String, BufferedWriter> entry : writerMap.entrySet()) {
            try {
                entry.getValue().close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        writerMap.clear();
        withStdOutMap.clear();
    }
}

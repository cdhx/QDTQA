package cn.edu.nju.ws.edgqa.domain.beans;

import org.jetbrains.annotations.NotNull;

import java.util.Objects;

public class Span {
    /**
     * The start position in the sequence.
     */
    private int begin;

    /**
     * The end position in the sequence.
     */
    private int end;

    /**
     * The string of this span [start, end)
     */
    private String str;

    public Span(int begin, int end, String str) {
        this.begin = begin;
        this.end = end;
        this.str = str;
    }

    /**
     * Determine if two spans have a overlap part
     *
     * @param span1 span 1
     * @param span2 span 2
     * @return if two spans have a overlap part, return true, false otherwise
     */
    public static boolean spanConflict(@NotNull Span span1, @NotNull Span span2) {
        return (span1.end > span2.begin && span2.end > span1.begin);
    }

    public int getBegin() {
        return begin;
    }

    public void setBegin(int begin) {
        this.begin = begin;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Span span = (Span) o;
        return begin == span.begin &&
                end == span.end &&
                str.equals(span.str);
    }

    @Override
    public int hashCode() {
        return Objects.hash(begin, end, str);
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public String getStr() {
        return str;
    }

    public void setStr(String str) {
        this.str = str;
    }

    public int getLength() {
        return this.end - this.begin + 1;
    }

    @Override
    public String toString() {
        return "Span{" +
                "start=" + begin +
                ", end=" + end +
                ", str='" + str + '\'' +
                '}';
    }

    /**
     * Determine if this span covers another span
     *
     * @param another another span
     * @return if this span covers another span, return true, false otherwise
     */
    public boolean covers(Span another) {
        return (begin <= another.begin && end >= another.end);
    }

    public boolean conflictsWith(@NotNull Span another) {
        return end > another.begin && another.end > begin;
    }
}

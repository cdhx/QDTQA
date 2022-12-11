package cn.edu.nju.ws.edgqa.domain.beans.tuple;

/**
 * A two-element tuple
 *
 * @author Saisai Gong
 */
public class TwoTuple<A, B> {
    private A first;
    private B second;

    public TwoTuple() {

    }

    public TwoTuple(A a, B b) {
        first = a;
        second = b;
    }

    public A getFirst() {
        return this.first;
    }

    public void setFirst(A a) {
        this.first = a;
    }

    public B getSecond() {
        return this.second;
    }

    public void setSecond(B b) {
        this.second = b;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((first == null) ? 0 : first.hashCode());
        result = prime * result + ((second == null) ? 0 : second.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        TwoTuple<A, B> other = (TwoTuple<A, B>) obj;
        if (first == null) {
            if (other.first != null)
                return false;
        } else if (!first.equals(other.first))
            return false;
        if (second == null) {
            if (other.second != null)
                return false;
        } else if (!second.equals(other.second))
            return false;
        return true;
    }

    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}

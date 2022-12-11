package cn.edu.nju.ws.edgqa.domain.beans;

import cn.edu.nju.ws.edgqa.utils.enumerates.LinkEnum;
import org.jetbrains.annotations.NotNull;

import java.io.Serializable;
import java.util.Objects;

public class Link implements Comparable, Serializable {
    /**
     * the mention in the natural language question
     */
    private String mention;

    /**
     * the linking uri
     */
    private String uri;

    /**
     * The type of this linking
     */
    private LinkEnum type;

    /**
     * the linking score
     */
    private double score;

    /**
     * the linking tool
     */
    private String matchedBy;

    /**
     * the mention in the original question, e.g., "38,49"
     */
    private String seq;

    /**
     * Constructor for golden linking
     *
     * @param mention   mention in the original natural language
     * @param uri       linking URI
     * @param type      linking type, entity, relation or type linking
     * @param matchedBy the linking method
     * @param seq       the begin and end position in the original sentence
     */
    public Link(String mention, String uri, LinkEnum type, String matchedBy, String seq) {
        this.mention = mention;
        this.uri = uri;
        this.type = type;
        this.matchedBy = matchedBy;
        this.seq = seq;
        this.score = 1.0;
    }

    /**
     * Constuctor, the default score is zero
     *
     * @param mention mention in the original natural language
     * @param uri     linking URI
     * @param type    linking type, entity, relation or type linking
     */
    public Link(String mention, String uri, LinkEnum type) {
        this.mention = mention;
        this.uri = uri;
        this.type = type;
        this.score = 0.0; //default score
    }

    public Link(String mention, String uri, LinkEnum type, double score) {
        this.mention = mention;
        this.uri = uri;
        this.type = type;
        this.score = score; //default score
    }

    public String getMention() {
        return mention;
    }

    public void setMention(String mention) {
        this.mention = mention;
    }

    public String getUri() {
        return uri;
    }

    public void setUri(String uri) {
        this.uri = uri;
    }

    public LinkEnum getType() {
        return type;
    }

    public void setType(LinkEnum type) {
        this.type = type;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public String getMatchedBy() {
        return matchedBy;
    }

    public void setMatchedBy(String matchedBy) {
        this.matchedBy = matchedBy;
    }

    public String getSeq() {
        return seq;
    }

    public void setSeq(String seq) {
        this.seq = seq;
    }

    @Override
    public String toString() {
        return "Link{" +
                "mention='" + mention + '\'' +
                ", uri='" + uri + '\'' +
                ", type=" + type +
                ", score=" + score +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Link link = (Link) o;
        return type == link.type &&
                mention.toLowerCase().equals(link.mention.toLowerCase()) &&
                uri.equals(link.uri);
    }

    @Override
    public int hashCode() {
        return Objects.hash(mention, uri, type);
    }


    @Override
    public int compareTo(@NotNull Object o) {
        if (o instanceof Link) {
            return Double.compare(this.getScore(), ((Link) o).getScore());
        }
        return 0;
    }
}

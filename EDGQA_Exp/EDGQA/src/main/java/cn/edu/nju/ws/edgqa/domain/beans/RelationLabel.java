package cn.edu.nju.ws.edgqa.domain.beans;

import org.jetbrains.annotations.NotNull;

/**
 * The relation label data for relation detection task.
 */
public class RelationLabel implements Comparable {
    /**
     * The original question
     */
    private String question;

    /**
     * The label of one relation uri in this question, note that a question may have mutliple relations
     */
    private String uri;

    /**
     * The mention text of this relation
     */
    private String mention;

    /**
     * 1 for true relation mention, 0 for false relation mention
     */
    private int score;

    public RelationLabel(String question, String uri, String mention, int score) {
        this.question = question;
        this.uri = uri;
        this.mention = mention;
        this.score = score;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public String getUri() {
        return uri;
    }

    public void setUri(String uri) {
        this.uri = uri;
    }

    public String getMention() {
        return mention;
    }

    public void setMention(String mention) {
        this.mention = mention;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public boolean equals(Object object) {
        if (!(object instanceof RelationLabel)) return false;

        RelationLabel relationLabel = (RelationLabel) object;
        if (score != relationLabel.getScore()) return false;
        if (!question.equals(relationLabel.getQuestion())) return false;
        if (!uri.equals(relationLabel.getUri())) return false;
        if (!mention.equals(relationLabel.getMention())) return false;
        return true;
    }

    public boolean keyEquals(RelationLabel relationLabel) {
        if (!question.equals(relationLabel.getQuestion())) return false;
        if (!uri.equals(relationLabel.getUri())) return false;
        if (!mention.equals(relationLabel.getMention())) return false;
        return true;
    }

    public String toFileLine() {
        return question + "\t" + uri + "\t" + mention + "\t" + score + "\r\n";
    }

    @Override
    public int compareTo(@NotNull Object o) {
        RelationLabel relationLabel = (RelationLabel) o;
        return toFileLine().compareTo(relationLabel.toFileLine());
    }
}
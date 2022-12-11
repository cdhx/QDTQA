package cn.edu.nju.ws.edgqa.domain.beans.relation_detection;

public class Paraphrase {

    /**
     * A predicate in KB
     */
    String predicate;

    /**
     * Paraphrase words
     */
    String paraphraseWords;

    /**
     * Similarity score
     */
    double score;

    public Paraphrase(String predicate, String paraphraseWords) {
        this.predicate = predicate;
        this.paraphraseWords = paraphraseWords;
        this.score = 1;
    }

    public Paraphrase(String predicate, String paraphraseWords, double score) {
        this.predicate = predicate;
        this.paraphraseWords = paraphraseWords;
        this.score = score;
    }

    public String getPredicate() {
        return predicate;
    }

    public void setPredicate(String predicate) {
        this.predicate = predicate;
    }

    public String getParaphraseWords() {
        return paraphraseWords;
    }

    public void setParaphraseWords(String paraphraseWords) {
        this.paraphraseWords = paraphraseWords;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }
}

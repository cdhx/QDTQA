package cn.edu.nju.ws.edgqa.domain.beans;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class TreeNode {
    /**
     * Data of this node
     */
    public String data;
    public String str;
    public List<TreeNode> children;
    public TreeNode parent;
    /**
     * Whether it is a leaf node
     */
    public boolean leaf;
    public int index;

    private TreeNode() {
        this.data = null;
        children = new ArrayList<>();
        parent = null;
        leaf = true;
        str = null;
        index = -1;
    }

    public TreeNode(String data) {
        this();
        this.data = data;
    }

    public static TreeNode createTree(@NotNull String s) {
        int x = 0;
        TreeNode root = new TreeNode();
        root.data = "ROOT ";
        String[] temp = s.split("\\(ROOT ");
        if (temp.length == 1) {
            temp = s.split("\\[ROOT ");
        }
        s = temp[1];
        char[] c = s.toCharArray();
        constructTree(root, s.length(), 0, c, 0);
        dealWithLeaves(root, 0);
        return root;
    }

    private static void constructTree(TreeNode root, int l, int i, char[] c, int x) {
        if (i == l) {
            return;
        }
        if (c[i] == '(' || c[i] == '[') {
            x++;
            TreeNode treeNode = new TreeNode();
            root.children.add(treeNode);
            root.leaf = false;
            treeNode.parent = root;
            int j = i + 1;
            StringBuilder temp = new StringBuilder();
            while (c[j] != '(' && c[j] != ')' && c[j] != '[' && c[j] != ']') {
                temp.append(c[j]);
                j++;
            }
            treeNode.data = temp.toString();
            constructTree(treeNode, l, j, c, x);
        } else if (c[i] == ')' || c[i] == ']') {
            x--;
            constructTree(root.parent, l, i + 1, c, x);
        } else {
            constructTree(root, l, i + 1, c, x);
        }
    }

    private static int dealWithLeaves(TreeNode root, int length) {
        if (root.children.size() == 0) {
            int index = root.data.indexOf(" ");
            root.str = root.data.substring(index);
            root.data = root.data.substring(0, index + 1);
            root.index = length;
            return length + 1;
        } else {
            for (int i = 0; i < root.children.size(); i++) {
                length = dealWithLeaves(root.children.get(i), length);
            }
            return length;
        }
    }

    public static boolean isContainLeafData(TreeNode treeNode, String s) {
        if (treeNode.isLeaf()) {
            if (treeNode.data.equals(s)) {
                return true;
            } else {
                return false;
            }
        } else {
            for (TreeNode treeNode1 : treeNode.children) {
                if (isContainLeafData(treeNode1, s)) {
                    return true;
                }
            }
            return false;
        }
    }

    public static boolean isContainLeafStr(TreeNode treeNode, String s) {
        if (treeNode.isLeaf()) {
            if (treeNode.str.equals(s)) {
                return true;
            } else {
                return false;
            }
        } else {
            for (TreeNode treeNode1 : treeNode.children) {
                if (isContainLeafStr(treeNode1, s)) {
                    return true;
                }
            }
            return false;
        }
    }

    public static TreeNode getFirstLeaf(TreeNode treeNode) {
        if (treeNode.isLeaf()) {
            return treeNode;
        } else return getFirstLeaf(treeNode.children.get(0));
    }

    /**
     * Get the most right leaf node of a treeNode
     *
     * @param treeNode the root of this tree
     * @return the most right leaf node of this tree
     */
    public static TreeNode getLastLeaf(TreeNode treeNode) {
        if (treeNode.isLeaf()) {
            return treeNode;
        } else return getLastLeaf(treeNode.children.get(treeNode.children.size() - 1));
    }

    /**
     * get the leaf node with a given index
     *
     * @param root  the root of the sub-tree
     * @param index the index of the leaf node
     * @return the leaf node with the given index, or null if it is not found
     */
    public static TreeNode getLeaf(TreeNode root, int index) {

        if (root.isLeaf()) { // current root is the leaf node, return it directly
            if (root.index == index) {
                return root;
            }
        }

        if (getFirstLeaf(root).index > index) { // the index of the most left leaf node is greater than index, return null
            return null;
        }
        for (TreeNode temp : root.children) { // keep looking in the children
            if (temp.isLeaf()) {
                if (temp.index == index) { // current node is the one we need
                    return temp;
                }
            } else if (getFirstLeaf(temp).index <= index && getLastLeaf(temp).index >= index) { // in the descendants of the current node
                return getLeaf(temp, index);
            }
        }
        return null;
    }

    public static List<TreeNode> FindAllLeafStr(@NotNull TreeNode root, String str, List<TreeNode> list) {
        if (root.isLeaf()) {
            if (root.str.equals(str)) {
                list.add(root);
            }
        } else {
            for (TreeNode temp : root.children) {
                list = FindAllLeafStr(temp, str, list);
            }
        }
        return list;
    }

    /**
     * Find the ancestor by start and end
     *
     * @param root  the root node of a tree
     * @param start the start index
     * @param end   the end index
     * @return the tree node of the ancestor
     */
    @Nullable
    public static TreeNode getAncestor(TreeNode root, int start, int end) {
        if (getFirstLeaf(root).index == start && getLastLeaf(root).index == end - 1) {
            return root;
        }
        for (int i = 0; i < root.children.size(); i++) {
            TreeNode treeNode = getAncestor(root.children.get(i), start, end);
            if (treeNode != null) {
                return treeNode;
            }
        }
        return null;
    }

    /**
     * Given a root node of a subtree and return the string of all the leaf nodes under the subtree
     *
     * @param treeNode the root of the sub-tree
     * @return a string concanated by leaf nodes of the sub-tree
     */
    public static String selectLeaf(TreeNode treeNode) {
        return selectLeaf(treeNode, "");
    }

    public static String selectLeaf(@NotNull TreeNode treeNode, String s) {
        //System.out.println(s+" "+treeNode.data);
        if (treeNode.leaf) {
            if (treeNode.str != null) {
                s = s + treeNode.str;
            } else {
                String[] t = treeNode.data.split(" ");
                treeNode.data = t[0];
                treeNode.str = t[1];
                s = s + " " + t[1];
            }
            return s;
        }
        for (int i = 0; i < treeNode.children.size(); i++) {
            s = selectLeaf(treeNode.children.get(i), s);
        }
        return s;
    }

    public static String selectLeafByIndex(TreeNode treeNode, int startIndex, int endIndex) {
        StringBuilder sb = new StringBuilder();
        for (int i = startIndex; i < endIndex; i++) {
            sb.append(getLeaf(treeNode, i).str.trim()).append(" ");
        }
        return sb.toString().trim();
    }

    public static String getLeafByIndex(TreeNode treeNode, int start, int end) {
        StringBuilder s = new StringBuilder();
        for (int i = start; i < end; i++) {
            s.append(getLeaf(treeNode, i).str.trim()).append(" ");
        }
        return s.toString().trim();
    }


    public static String getSubString(TreeNode treeNode) {
        LinkedList<TreeNode> stack = new LinkedList<>();
        stack.push(treeNode);
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            TreeNode current = stack.pop();
            if (current.isLeaf()) {
                sb.append(current.str.trim()).append(" ");
            } else {
                List<TreeNode> childs = current.getChildren();
                ListIterator<TreeNode> iter = childs.listIterator(childs.size());
                while (iter.hasPrevious()) {
                    stack.push(iter.previous());
                }
            }
        }
        return sb.toString().trim();
    }


    /**
     * Get the substring of nodes before the bound index
     *
     * @param treeNode a root node of a tree
     * @param bound    the bound index
     * @return the substring
     */
    public static String getBoundedSubString(@NotNull TreeNode treeNode, int bound) {
        List<TreeNode> childs = treeNode.getChildren();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bound; i++) {
            sb.append(selectLeaf(childs.get(i), "")).append(" ");
        }
        return sb.toString().trim();
    }


    public static boolean isFirstLayerContainStr(@NotNull TreeNode treeNode, String s) {
        for (int j = 0; j < treeNode.children.size(); j++) {
            TreeNode treechild = treeNode.children.get(j);
            while (!treechild.isLeaf() && treechild.children.size() == 1) {
                treechild = treechild.children.get(0);
            }
            if (treechild.isLeaf()) {
                if (treechild.str.substring(1).equals(s)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Get the most left leaf node
     *
     * @return the most left leaf node if available, null otherwise
     */
    public TreeNode getFirstChild() {
        if (this.isLeaf()) {
            return null;
        } else {
            return this.getChildren().get(0);
        }
    }

    /**
     * Get the most right leaf node
     *
     * @return the most right leaf node if available, null otherwise
     */
    public TreeNode getLastChild() {
        if (this.isLeaf()) {
            return null;
        } else {
            return this.getChildren().get(this.getChildren().size() - 1);
        }
    }


    public String getData() {
        return data;
    }

    public void setData(String data) {
        this.data = data;
    }

    public String getStr() {
        return str;
    }

    public void setStr(String str) {
        this.str = str;
    }

    public List<TreeNode> getChildren() {
        return children;
    }

    public void setChildren(List<TreeNode> children) {
        this.children = children;
    }

    public TreeNode getParent() {
        return parent;
    }

    public void setParent(TreeNode parent) {
        this.parent = parent;
    }

    public boolean isLeaf() {
        return leaf;
    }

    public void setLeaf(boolean leaf) {
        this.leaf = leaf;
    }

    @Override
    public String toString() {
        return "Tree { data='" + data + '\'' +
                ", str='" + str + '\'' +
                ", leaf=" + leaf +
                ", index=" + index +
                '}';
    }
}

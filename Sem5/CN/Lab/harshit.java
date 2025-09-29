import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class harshit {
    private static boolean findAPath(List<List<Integer>> obstacles, List<List<Integer>> portals, int n, int m, int x, int y, int fx, int fy) {
        if(x >= n || y >= m) return false;
        if(x == fx && y == fy) return true;

        for(List<Integer> obstacle : obstacles) {
            //check if right is obsatcle
            if(obstacle.get(0) == x+1 && obstacle.get(1) == y) {
                //if right is obstacle check if down is obstacle
                for(List<Integer> obstacle2 : obstacles) {
                    if(obstacle2.get(0) == x && obstacle2.get(1) == y+1)
                        return false;
                }
                //if down is not obstacle check if down is a portal
                for(List<Integer> portal : portals) {
                    if(portal.get(0) == x && portal.get(1) == y+1)
                        return findAPath(obstacles, portals, n, m, portal.get(2), portal.get(3), fx, fy);
                }
                //if right is obstacle and down is neither a portal not an obstacle then down is the only path
                return findAPath(obstacles, portals, n, m, x, y+1, fx, fy);
            }
        }
        //if right is not an obstacle check if its a portal
        for(List<Integer> portal : portals) {
            if(portal.get(0) == x+1 && portal.get(1) == y)
                return findAPath(obstacles, portals, n, m, portal.get(2), portal.get(3), fx, fy);
        }
        //if right is neither an obstacle nor a portal its a path
        return findAPath(obstacles, portals, n, m, x+1, y, fx, fy); 
    }

    public static void main(String[] args) {
        int n = 5, m = 6;
        // Portals: each portal is [x1, y1, x2, y2]
        List<List<Integer>> portals = new ArrayList<>();
        portals.add(Arrays.asList(1, 0, 3, 4));
        portals.add(Arrays.asList(2, 2, 4, 5));
        portals.add(Arrays.asList(2, 0, 4, 0));

        // Obstacles: each obstacle is [row, col]
        List<List<Integer>> obstacles = new ArrayList<>();
        obstacles.add(Arrays.asList(0, 3));
        obstacles.add(Arrays.asList(2, 1));
        obstacles.add(Arrays.asList(4, 4));

        System.out.println(findAPath(obstacles, portals, n, m, 0, 0, 3, 4));
    }
}

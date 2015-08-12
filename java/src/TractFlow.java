import java.io.*;
import java.util.HashMap;

/**
 * Created by Hongjian on 8/12/2015.
 */
public class TractFlow {

    public TractFlow() {
        // do nothing
    }

    static int[] mergeBlockCensus(int[] a, int[] b) {
        if (a.length != b.length)
            System.out.println("Two lists' length don't mathc!");
        for (int i = 0; i < a.length; i++)
            a[i] += b[i];
        return a;
    }


    public static void main(String[] args) {
        HashMap<Long, HashMap<Long, int[]>> tracts = new HashMap<>();
        String dirPath = "../data/2010/";
        File folder = new File(dirPath);
        String[] fnames = folder.list();

        int cnt = 0;
        for (String fn : fnames) {
            try (BufferedReader br = new BufferedReader(new FileReader(dirPath + fn))) {
                String l;
                br.readLine();  // get rid of header
                while ((l = br.readLine())!= null) {
                    String[] ls = l.split(",");
//                    String org = ls[0].substring(0, ls[0].length()-3);
//                    String dst = ls[1].substring(0, ls[1].length()-3);
                    long org = Long.parseLong(ls[0]) / 1000;
                    long dst = Long.parseLong(ls[1]) / 1000;
                    int[] counts = new int[10];
                    for (int i = 0; i < 10; i++)
                        counts[i] = Integer.parseInt(ls[i+2]);

                    if (tracts.containsKey(org)) {
                        if (tracts.get(org).containsKey(dst)) {
                            int[] old_counts = tracts.get(org).get(dst);
                            tracts.get(org).put(dst, mergeBlockCensus(old_counts, counts));
                        } else {
                            tracts.get(org).put(dst, counts);
                        }
                    } else {
                        tracts.put(org, new HashMap<>());
                        tracts.get(org).put(dst, counts);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            cnt++;
            if (cnt % 5 == 0)
                System.out.printf("%d out of %d files processed.\n", cnt, fnames.length);
        }


        try (BufferedWriter bw = new BufferedWriter(new FileWriter("../data/state_all_tract_level_od_JT00_2010")) ) {
            for (long org : tracts.keySet()) {
                bw.write(Long.toString(org) + ",");
                for (long dst : tracts.get(org).keySet()) {
                    bw.write(Long.toString(dst));
                    for (int val : tracts.get(org).get(dst)) {
                        bw.write("," + Integer.toString(val));
                    }
                    bw.write("\n");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}

package FatSLIM;
import net.librec.conf.Configuration;
import net.librec.job.RecommenderJob;
import net.librec.math.algorithm.Randoms;

import java.io.FileInputStream;
import java.util.Properties;


public class Driver {

    // Change this to load a different configuration file.
    public static String CONFIG_FILE = "/Users/nasimsonboli/IdeaProjects/FAW-RS/conf/ItemSLIM_Kiva.properties";

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        String confFilePath = CONFIG_FILE;
        Properties prop = new Properties();
        prop.load(new FileInputStream(confFilePath));
        for (String name : prop.stringPropertyNames()) {
            conf.set(name, prop.getProperty(name));
        }

        Randoms.seed(20170701);
        RecommenderJob job = new RecommenderJob(conf);
        job.runJob();
        System.out.print("Finished");
    }
}

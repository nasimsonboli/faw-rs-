package FatSLIM;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.AbstractRecommender;
import net.librec.util.Lists;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


/**
 * This work is heavily borrowed from
 * Xia Ning and George Karypis, <strong>SLIM: Sparse Linear Methods for Top-N Recommender Systems</strong>, ICDM 2011. <br>
 * except that this method is used for User SLIM instead of item SLIM and
 * The objective function includes user balance as well to achieve parity.
 * written by Nasim Sonboli
 */
@ModelData({"isRanking", "slim", "coefficientMatrix", "trainMatrix", "similarityMatrix", "knn"})
public class Slim_FatItems_Recommender extends AbstractRecommender {
    /**
     * the number of iterations
     */
    protected int numIterations;

    /**
     * W in original paper, a sparse matrix of aggregation coefficients
     */
    private DenseMatrix coefficientMatrix;

    /**
     * item's nearest neighbors for kNN > 0
     */
    private Set<Integer>[] itemNNs;

    /**
     * regularization parameters for the L1 or L2 term
     */
    private float regL1Norm, regL2Norm;

    /**
     *This parameter controls the influence of item balance calculation on the overall optimization.
     */
    private float Lambda3;

    /**
     * This vector is a 1 x M vector, and M is the number of users,
     * this vector is filled with either 1 or -1,
     * If a user belongs to the protected group it is +1, otherwise it is -1
     */
    private double[] groupMembershipVector;

    /**
     * balance
     */
//    private static double balance;

    /**
     * number of nearest neighbors
     */
    protected static int knn;

    /**
     * item similarity matrix
     */
    private SymmMatrix similarityMatrix;

    /**
     * items's nearest neighbors for kNN <= 0, i.e., all other items
     */
    private Set<Integer> allItems;

    /**
     * reading the membership file
     */
    private String membershipFilePath;

    /**
     * initialization
     *
     * @throws LibrecException if error occurs
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        knn = conf.getInt("rec.neighbors.knn.number", 50);
        numIterations = conf.getInt("rec.iterator.maximum");
        regL1Norm = conf.getFloat("rec.slim.regularization.l1", 1.0f);
        regL2Norm = conf.getFloat("rec.slim.regularization.l2", 1.0f);
        Lambda3 = conf.getFloat("rec.slim.regularization.item.balance.controller", 1.0f);
        membershipFilePath = conf.get("data.membership.input.path", "/data/membership/ml-1m/");
        // set it in configuration file

        System.out.println("***");
        System.out.println(regL1Norm);
        System.out.println(regL2Norm);
        System.out.println(Lambda3);
        System.out.println("***");

        coefficientMatrix = new DenseMatrix(numItems, numItems);
        // initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed up training
        coefficientMatrix.init();
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
        System.out.println("Done with the similarity Matrix...");


        for(int itemIdx = 0; itemIdx < this.numItems; ++itemIdx) {
            this.coefficientMatrix.set(itemIdx, itemIdx, 0.0d);
        } //iterate through all of the items , initialize

        //create the nn matrix
        createItemNNs();

        //reading all the files in a directory, because we only have 1 file, we read filenames[0], can be extended
        File[] filenames = new File(membershipFilePath).listFiles();
        System.out.println("Reading file: " + filenames[0]);
        String content = readFile(filenames[0]);
        String[] userSplited = content.split("\n");

        groupMembershipVector = new double[numUsers];
        //fill in the membership vector by membership numbers (1, -1)
        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            //each row contains itemIdx and membership, take only the membership number
            int membership = Integer.parseInt(userSplited[itemIdx].split("\t")[1]);
            groupMembershipVector[itemIdx] = membership;
        }
        System.out.println("Done Reading membership file...!");
    }

    /**
     * Reading a file
     */
    public String readFile(File file)
    {
        String content = null;
        FileReader reader = null;
        try {
            reader = new FileReader(file);
            char[] chars = new char[(int) file.length()];
            reader.read(chars);
            content = new String(chars);
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if(reader !=null){reader.close();} } catch (IOException ie) {ie.printStackTrace();}
        }
        return content;
    }

    /**
     * train model
     *
     * @throws LibrecException if error occurs
     */
    @Override
    protected void trainModel() throws LibrecException {
        // number of iteration cycles
        for (int iter = 1; iter <= numIterations; iter++) {

            loss = 0.0d;
            // each cycle iterates through one coordinate direction
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                // find k-nearest neighbors of each item
                Set<Integer> nearestNeighborCollection = knn > 0 ? itemNNs[itemIdx] : allItems;

                //all the ratings for itemIdx from all the users
                double[] userRatingEntries = new double[numUsers];

                Iterator<VectorEntry> userItr = trainMatrix.rowIterator(itemIdx); //look for the ratings of all users for one item
                while (userItr.hasNext()) {
                    VectorEntry userRatingEntry = userItr.next();
                    userRatingEntries[userRatingEntry.index()] = userRatingEntry.get();
                }

                // for each nearest neighbor nearestNeighborItemIdx, update coefficient Matrix by the coordinate
                // descent update rule
                for (Integer nearestNeighborItemIdx : nearestNeighborCollection) { //user nearest neighbors
                    if (nearestNeighborItemIdx != itemIdx) {
                        double gradSum = 0.0d, rateSum = 0.0d, errors = 0.0d, itemBalanceSum =0.0d;

                        //ratings of each item for all the other users
                        Iterator<VectorEntry> nnUserRatingItr = trainMatrix.rowIterator(nearestNeighborItemIdx);
                        if (!nnUserRatingItr.hasNext()) {
                            continue;
                        }

                        int nnCount = 0;

                        while (nnUserRatingItr.hasNext()) { // now go through the ratings a user has put on items
                            VectorEntry nnUserVectorEntry = nnUserRatingItr.next();
                            int nnUserIdx = nnUserVectorEntry.index();
                            double nnRating = nnUserVectorEntry.get();
                            double rating = userRatingEntries[nnUserIdx]; //get the rating of the nn user on the main item!!
                            double error = rating - predict(nnUserIdx, itemIdx, nearestNeighborItemIdx);


                            // Calculating Sigma(pk . wik)
                            double itembalance = balancePredictor(nnUserIdx, itemIdx, nearestNeighborItemIdx);



                            itemBalanceSum += itembalance * itembalance; //item balance squared
                            gradSum += nnRating * error;
                            rateSum += nnRating * nnRating; // sigma r^2

                            errors += error * error;
                            nnCount++;
                        }

                        itemBalanceSum /= nnCount; // Doubt: user balance sum (?) why should we divide it by nnCount?
                        gradSum /= nnCount;
                        rateSum /= nnCount;

                        errors /= nnCount;



                        double coefficient = coefficientMatrix.get(nearestNeighborItemIdx, itemIdx);
                        double nnMembership = groupMembershipVector[itemIdx];
                        // Loss function
                        loss += 0.5 * errors + 0.5 * regL2Norm * coefficient * coefficient + regL1Norm * coefficient + 0.5 * Lambda3 * itemBalanceSum ;


                        /** Implementing Soft Thresholding => S(beta, Lambda1)+
                         * beta = Sigma(r - Sigma(wr)) + Lambda3 * p * Sigma(wp)
                         * & Sigma(r - Sigma(wr)) = gradSum
                         * & nnMembership = p
                         * & Sigma(wp) = userBalanceSum
                         */
                        double beta = gradSum + (Lambda3 * nnMembership * itemBalanceSum) ; //adding user balance to the gradsum
                        double update = 0.0d; //weight

                        if (regL1Norm < Math.abs(beta)) {
                            if (beta > 0) {
                                update = (beta - regL1Norm) / (regL2Norm + rateSum + Lambda3);
                            } else {
                                // One doubt: in this case, wij<0, however, the
                                // paper says wij>=0. How to gaurantee that?
                                update = (beta + regL1Norm) / (regL2Norm + rateSum + Lambda3);
                            }
                        }

                        coefficientMatrix.set(nearestNeighborItemIdx, itemIdx, update); //update the coefficient
                    }
                }
            }

            if (isConverged(iter) && earlyStop) {
                break;
            }
        }
    }

    /**
     * predict a specific ranking score for user userIdx on item itemIdx.
     *
     * @param userIdx         user index
     * @param itemIdx         item index
     * @param excludedItemIdx excluded item index
     * @return a prediction without the contribution of excluded item
     */
    protected double predict(int userIdx, int itemIdx, int excludedItemIdx) {

        double predictRating = 0;
        Iterator<VectorEntry> itemEntryIterator = trainMatrix.colIterator(userIdx);

        while (itemEntryIterator.hasNext()) {
            VectorEntry itemEntry = itemEntryIterator.next();
            int nearestNeighborItemIdx = itemEntry.index();
            double nearestNeighborPredictRating = itemEntry.get();

            if (itemNNs[itemIdx].contains(nearestNeighborItemIdx) && nearestNeighborItemIdx != excludedItemIdx) {

                double coeff = coefficientMatrix.get(nearestNeighborItemIdx, itemIdx);
                predictRating += nearestNeighborPredictRating * coeff;
            }
        }
        return predictRating;
    }
    /**
     * calculate the balance for each item according to their membership weight and their coefficient
     *  diag(PW) ^ 2
     *  for all of the nnItems of an item
     */
    protected double balancePredictor(int userIdx, int itemIdx, int excludedItemIdx) {

        double predictBalance = 0;
        Iterator<VectorEntry> itemEntryIterator = trainMatrix.colIterator(userIdx);
        while (itemEntryIterator.hasNext()) { //iterate through the nearest neighbors of an item and calculate the prediction accordingly
            VectorEntry itemEntry = itemEntryIterator.next();
            int nearestNeighborItemIdx = itemEntry.index(); //nn item index

            if (itemNNs[itemIdx].contains(nearestNeighborItemIdx) && nearestNeighborItemIdx != excludedItemIdx) {
                //take p vector, multiply by the coefficients of neighbors (dot product)
                predictBalance += groupMembershipVector[nearestNeighborItemIdx] * coefficientMatrix.get(nearestNeighborItemIdx, itemIdx);
            }
        }
        return predictBalance;
    }


    @Override
    protected boolean isConverged(int iter) {
        double delta_loss = lastLoss - loss;
        lastLoss = loss;

        // print out debug info
        if (verbose) {
            String recName = getClass().getSimpleName().toString();
            String info = recName + " iter " + iter + ": loss = " + loss + ", delta_loss = " + delta_loss;
            LOG.info(info);
        }

        return iter > 1 ? delta_loss < 1e-5 : false;
    }


    /**
     * predict a specific ranking score for user userIdx on item itemIdx.
     *
     * @param userIdx user index
     * @param itemIdx item index
     * @return predictive ranking score for user userIdx on item itemIdx
     * @throws LibrecException if error occurs
     */
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
//        create item knn list if not exists,  for local offline model
        if (!(null != itemNNs && itemNNs.length > 0)) {
            createItemNNs();
        }
        return predict(userIdx, itemIdx, -1);
    }


    /**
     * Create item KNN list.
     */
    public void createItemNNs() {
        itemNNs = new HashSet[numItems];

        // find the nearest neighbors for each item based on item similarity
        List<Map.Entry<Integer, Double>> tempItemSimList;
        if (knn > 0) {
            for (int itemIdx = 0; itemIdx < numItems; ++itemIdx) {
                SparseVector similarityVector = similarityMatrix.row(itemIdx);
                if (knn < similarityVector.size()) {
                    tempItemSimList = new ArrayList<>(similarityVector.size() + 1);
                    Iterator<VectorEntry> simItr = similarityVector.iterator();
                    while (simItr.hasNext()) {
                        VectorEntry simVectorEntry = simItr.next();
                        tempItemSimList.add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
                    }
                    tempItemSimList = Lists.sortListTopK(tempItemSimList, true, knn);
                    itemNNs[itemIdx] = new HashSet<>((int) (tempItemSimList.size() / 0.5));
                    for (Map.Entry<Integer, Double> tempItemSimEntry : tempItemSimList) {
                        itemNNs[itemIdx].add(tempItemSimEntry.getKey());
                    }
                } else {
                    itemNNs[itemIdx] = similarityVector.getIndexSet();
                }
            }
        } else {
            allItems = new HashSet<>(trainMatrix.columns());
        }
    }
}

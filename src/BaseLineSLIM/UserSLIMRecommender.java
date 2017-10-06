package BaseLineSLIM;

/**
 * Created by Nasim on 6/17/2017.
 */

//package src.main.java.net.librec.recommender.cf.ranking;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.AbstractRecommender;
import net.librec.util.Lists;

import java.util.*;

//package net.librec.recommender.cf.ranking;


/**
 * Xia Ning and George Karypis, <strong>SLIM: Sparse Linear Methods for Top-N Recommender Systems</strong>, ICDM 2011. <br>
 * <p>
 * Related Work:
 * <ul>
 * <li>Levy and Jack, Efficient Top-N Recommendation by Linear Regression, ISRS 2013. This paper reports experimental
 * results on the MovieLens (100K, 10M) and Epinions datasets in terms of precision, MRR and HR@N (i.e., Recall@N).</li>
 * <li>Friedman et al., Regularization Paths for Generalized Linear Models via Coordinate Descent, Journal of
 * Statistical Software, 2010.</li>
 * </ul>
 *
 * @author guoguibing and Keqiang Wang
 */
@ModelData({"isRanking", "slim", "coefficientMatrix", "trainMatrix", "similarityMatrix", "knn"})
public class UserSLIMRecommender extends AbstractRecommender {
    /**
     * the number of iterations
     */
    protected int numIterations;

    /**
     * W in original paper, a sparse matrix of aggregation coefficients
     */
    private DenseMatrix coefficientMatrix;

    /**
     * user's nearest neighbors for kNN > 0
     */
    private Set<Integer>[] userNNs;

    /**
     * regularization parameters for the L1 or L2 term
     */
    private float regL1Norm, regL2Norm;

    /**
     * number of nearest neighbors
     */
    protected static int knn;

    /**
     * item similarity matrix
     */
    private SymmMatrix similarityMatrix;

    /**
     * users's nearest neighbors for kNN <=0, i.e., all other items
     */
    private Set<Integer> allUsers;

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
        // set it in configuration file

//        coefficientMatrix = new DenseMatrix(numItems, numItems);
        coefficientMatrix = new DenseMatrix(numUsers, numUsers);
        // initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed up training
        coefficientMatrix.init();
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();


        for(int userIdx = 0; userIdx < this.numUsers; ++userIdx) {
            this.coefficientMatrix.set(userIdx, userIdx, 0.0d);
        } //iterate through all of the users , initialize

        createUserNNs();
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
            for (int userIdx = 0; userIdx < numUsers; userIdx++) {
                // find k-nearest neighbors of each user
                Set<Integer> nearestNeighborCollection = knn > 0 ? userNNs[userIdx] : allUsers;

                //all the ratings of userIdx for all the items
                double[] itemRatingEntries = new double[numItems];

                Iterator<VectorEntry> itemItr = trainMatrix.colIterator(userIdx);
                while (itemItr.hasNext()) {
                    VectorEntry itemRatingEntry = itemItr.next();
                    itemRatingEntries[itemRatingEntry.index()] = itemRatingEntry.get();
                }

                // for each nearest neighbor nearestNeighborItemIdx, update coefficienMatrix by the coordinate
                // descent update rule
                for (Integer nearestNeighborUserIdx : nearestNeighborCollection) { //user nearest neighbors!
                    if (nearestNeighborUserIdx != userIdx) {
                        double gradSum = 0.0d, rateSum = 0.0d, errors = 0.0d;

                        //ratings of each user on all the other items
                        //System.out.print(nearestNeighborUserIdx); //
                        Iterator<VectorEntry> nnItemRatingItr = trainMatrix.colIterator(nearestNeighborUserIdx);
                        if (!nnItemRatingItr.hasNext()) {
                            continue;
                        }

                        int nnCount = 0;

                        while (nnItemRatingItr.hasNext()) {
                            VectorEntry nnItemVectorEntry = nnItemRatingItr.next();
                            int nnItemIdx = nnItemVectorEntry.index();
                            double nnRating = nnItemVectorEntry.get();
                            double rating = itemRatingEntries[nnItemIdx];
                            double error = rating - predict(userIdx, nnItemIdx, nearestNeighborUserIdx);

                            gradSum += nnRating * error;
                            rateSum += nnRating * nnRating;

                            errors += error * error;
                            nnCount++;
                        }


                        gradSum /= nnCount;
                        rateSum /= nnCount;

                        errors /= nnCount;

                        double coefficient = coefficientMatrix.get(nearestNeighborUserIdx, userIdx);//nasim fixed
//                        loss += errors + 0.5 * regL2Norm * coefficient * coefficient + regL1Norm * coefficient; //Nasim edited
                        loss += 0.5 * errors + 0.5 * regL2Norm * coefficient * coefficient + regL1Norm * coefficient;


                        double update = 0.0d;
                        if (regL1Norm < Math.abs(gradSum)) {
                            if (gradSum > 0) {

                                update = (gradSum - regL1Norm) / (regL2Norm + rateSum);
                            } else {
                                // One doubt: in this case, wij<0, however, the
                                // paper says wij>=0. How to gaurantee that?
                                update = (gradSum + regL1Norm) / (regL2Norm + rateSum);
                            }
                        }

                        coefficientMatrix.set(nearestNeighborUserIdx, userIdx, update);//update the coefficient
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
     * @param excludedUserIdx excluded user index
     * @return a prediction without the contribution of excluded item
     */
    protected double predict(int userIdx, int itemIdx, int excludedUserIdx) {
        double predictRating = 0;
        Iterator<VectorEntry> userEntryIterator = trainMatrix.rowIterator(itemIdx);
        while (userEntryIterator.hasNext()) {
            VectorEntry userEntry = userEntryIterator.next();
            int nearestNeighborUserIdx = userEntry.index(); //nn user
            double nearestNeighborPredictRating = userEntry.get();
            if (userNNs[userIdx].contains(nearestNeighborUserIdx) && nearestNeighborUserIdx != excludedUserIdx) {
                predictRating += nearestNeighborPredictRating * coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
            }
        }

        return predictRating;
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
        if (!(null != userNNs && userNNs.length > 0)) {
            createUserNNs();
        }
        return predict(userIdx, itemIdx, -1);
    }


    /**
     * Create user KNN list.
     */
    public void createUserNNs() {
        userNNs = new HashSet[numUsers];

        // find the nearest neighbors for each user based on user similarity???
        List<Map.Entry<Integer, Double>> tempUserSimList;
        if (knn > 0) {
            for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
                SparseVector similarityVector = similarityMatrix.row(userIdx);
                if (knn < similarityVector.size()) {
                    tempUserSimList = new ArrayList<>(similarityVector.size() + 1);
                    Iterator<VectorEntry> simItr = similarityVector.iterator();
                    while (simItr.hasNext()) {
                        VectorEntry simVectorEntry = simItr.next();
                        tempUserSimList.add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
                    }
                    tempUserSimList = Lists.sortListTopK(tempUserSimList, true, knn);
                    userNNs[userIdx] = new HashSet<>((int) (tempUserSimList.size() / 0.5)); // why 0.5??
                    for (Map.Entry<Integer, Double> tempUserSimEntry : tempUserSimList) {
                        userNNs[userIdx].add(tempUserSimEntry.getKey());
                    }
                } else {
                    userNNs[userIdx] = similarityVector.getIndexSet();
                }
            }
        } else {
            allUsers = new HashSet<>(trainMatrix.columns());
        }
    }
}

#include "util/Base.h"
#include "util/Conf.h"
#include "structs/Matrix.h"
#include "util/Logger.h"
#include "alg/Naive.h"
#include "alg/tree/BallTreeSearch.h"
#include "alg/svd/SVDIncrPrune.h"
#include "alg/svd/SVDIntUpperBoundIncrPrune.h"
#include "alg/svd/SVDIntUpperBoundIncrPrune2.h"
#include "alg/int/IntUpperBound.h"
#include "alg/tree/FastMKS.h"
#include "alg/transformation/TransformIncrPrune.h"
#include "alg/simd/SIMDIntUpperBound.h"
#include "alg/svd/SVDIncrPruneIndividualReorder.h"
#include "alg/transformation/TransformSVDIncrPrune.h"
#include "alg/transformation/TransformSVDIncrPrune2.h"
#include "alg/int/IntUpperBound2.h"
#include <boost/program_options.hpp>
#include <omp.h>

#include <random>
#include <algorithm>
#include <cblas.h>

#define L2_CACHE_SIZE 256000
#define MAX_MEM_SIZE (257840L*1024L*1024L)

namespace po = boost::program_options;

void basicLog(const Matrix &q, const Matrix &p, const int k) {
  Logger::Log("q path: " + to_string(Conf::qDataPath));
  Logger::Log("p path: " + to_string(Conf::pDataPath));
  Logger::Log("q: " + to_string(q.rowNum) + "," + to_string(q.colNum));
  Logger::Log("p: " + to_string(p.rowNum) + "," + to_string(p.colNum));
  Logger::Log("Algorithm: " + Conf::algName);
  Logger::Log("k: " + to_string(k));
}

inline void computeTopRating(double *ratings_matrix, int *top_K_items,
                             const int num_users, const int num_items) {
  for (int user_id = 0; user_id < num_users; user_id++) {

    unsigned long index = user_id;
    index *= num_items;
    int best_item_id = cblas_idamax(num_items, &ratings_matrix[index], 1);
    top_K_items[user_id] = best_item_id;
  }
}

inline void computeTopK(double *ratings_matrix, int *top_K_items,
                        const int num_users, const int num_items, const int K) {

  for (int i = 0; i < num_users; i++) {

    // TODO: allocate vector on the stack, reserve the size we need or use the
    // insertion-and-copy array strategy that Matei suggested
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int> >,
                        std::greater<std::pair<double, int> > > q;

    unsigned long index = i;
    index *= num_items;

    for (int j = 0; j < K; j++) {
      q.push(std::make_pair(ratings_matrix[index + j], j));
    }

    for (int j = K; j < num_items; j++) {
      if (ratings_matrix[index + j] > q.top().first) {
        q.pop();
        q.push(std::make_pair(ratings_matrix[index + j], j));
      }
    }

    for (int j = 0; j < K; j++) {
      const std::pair<double, int> p = q.top();
      top_K_items[i * K + K - 1 - j] = p.second;
      q.pop();
    }
  }
}

inline double decisionRuleBlockedMM(Matrix &q, Matrix &p,
                                    const unsigned int rand_ind,
                                    const unsigned long num_users_per_block) {

  Monitor tt;
  double *user_ptr = q.getRowPtr(rand_ind);
  double *item_ptr = p.getRowPtr(0);
  const long m = num_users_per_block;
  const int n = p.rowNum;
  const int k = q.colNum;
  const float alpha = 1.0;
  const float beta = 0.0;
  double *matrix_product = (double *)malloc(m * n * sizeof(double));
  int *top_K_items = (int *)malloc(m * Conf::k * sizeof(int));

  tt.start();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, user_ptr,
              k, item_ptr, k, beta, matrix_product, n);

  if (Conf::k == 1) {
    computeTopRating(matrix_product, top_K_items, m, n);
  } else {
    computeTopK(matrix_product, top_K_items, m, n, Conf::k);
  }
  tt.stop();
  free(matrix_product);
  free(top_K_items);
  return tt.getElapsedTime();
}

int main(int argc, char **argv) {
  omp_set_dynamic(0);
  omp_set_num_threads(1);

  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "alg", po::value<string>(&(Conf::algName))->default_value("naive"),
      "Algorithm")("k", po::value<int>(&(Conf::k))->default_value(1),
                   "K")("dataset", po::value<string>(&(Conf::dataset)),
                        "name of dataset for log output")(
      "q", po::value<string>(&(Conf::qDataPath)), "file path of q Data")(
      "p", po::value<string>(&(Conf::pDataPath)), "file path of p Data")(
      "scalingValue", po::value<int>(&(Conf::scalingValue))->default_value(127),
      "maximum value for scaling")(
      "SIGMA", po::value<double>(&(Conf::SIGMA))->default_value(0.8),
      "percentage of SIGMA for SVD incremental prune")(
      "log", po::value<bool>(&(Conf::log))->default_value(true),
      "whether it outputs log")(
      "logPathPrefix",
      po::value<string>(&(Conf::logPathPrefix))->default_value("./log"),
      "output path of log file (Prefix)")(
      "outputResult",
      po::value<bool>(&(Conf::outputResult))->default_value(true),
      "whether it outputs results")(
      "resultPathPrefix",
      po::value<string>(&(Conf::resultPathPrefix))->default_value("./result"),
      "output path of result file (Prefix)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 0;
  } else if (Conf::qDataPath == "" || Conf::pDataPath == "") {
    cout << "Please specify path to data files" << endl << endl;
    cout << desc << endl;
    return 0;
  }

  //    Conf::qDataPath = "../../data/MovieLens/q.txt";
  //    Conf::pDataPath = "../../data/MovieLens/p.txt";
  //    Conf::dataset = "MovieLens";
  //    Conf::k = 1;
  //    Conf::SIGMA = 0.7;
  //    Conf::algName = "FEIPR-I2";
  //    Conf::algName = "FEIPR-I";
  //    Conf::algName = "FEIPR-S";

  Conf::Output();

  Matrix q;
  Matrix p;
  q.readData(Conf::qDataPath);
  p.readData(Conf::pDataPath);
  Conf::dimension = p.colNum;

  cout << "-----------------------" << endl;

  // ToDo: replace the old name (FEIPR) with FEXIPRO

  if (Conf::algName == "Naive") {
    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);

    naive(Conf::k, q, p);

  } else if (Conf::algName == "SIR") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::scalingValue) + "-" +
                         to_string(Conf::SIGMA) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("SIGMA: " + to_string(Conf::SIGMA));
    Logger::Log("Scaling Value: " + to_string(Conf::scalingValue));

    // Construct index
    SIRPrune sirPrune(Conf::k, Conf::scalingValue, Conf::SIGMA, &q, &p);

#ifdef ONLINE_DECISION_RULE
    std::random_device rd; // only used once to initialise (seed) engine
    std::mt19937 rng(
        rd()); // random-number engine used (Mersenne-Twister in this case)
    unsigned long num_users_per_block =
        4 * (L2_CACHE_SIZE / (sizeof(double) * q.colNum));
    while (num_users_per_block*p.rowNum*sizeof(double) > MAX_MEM_SIZE) {
      num_users_per_block /= 2;
    }
    std::uniform_int_distribution<int> uni(
        0, q.rowNum - num_users_per_block); // guaranteed unbiased
    const unsigned int rand_ind = uni(rng);

    const double blocked_mm_time =
        decisionRuleBlockedMM(q, p, rand_ind, num_users_per_block);

    Monitor tt;
    tt.start();
    sirPrune.topK(rand_ind, rand_ind + num_users_per_block);
    tt.stop();

    const double fexipro_time = tt.getElapsedTime();

    Logger::Log("Blocked MM time: " + to_string(blocked_mm_time));
    Logger::Log("FEXIPRO time: " + to_string(fexipro_time));
    if (blocked_mm_time < fexipro_time) {
      Logger::Log("Blocked MM wins");
    } else {
      Logger::Log("FEXIPRO wins");
#ifndef TEST_ONLY
      sirPrune.topK(0, rand_ind);
      sirPrune.topK(rand_ind + num_users_per_block, q.rowNum);
      sirPrune.addToOnlineTime(blocked_mm_time);
      sirPrune.outputResults();
#endif
    }
#else
    sirPrune.topK(0, q.rowNum);
    sirPrune.outputResults();
#endif

  } else if (Conf::algName == "SR") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::scalingValue) + "-" +
                         to_string(Conf::SIGMA) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("SIGMA: " + to_string(Conf::SIGMA));

    TransformSVDIncrPrune2 transformSVDIncrPrune2(Conf::k, Conf::SIGMA, &q, &p);
    transformSVDIncrPrune2.topK();

  } else if (Conf::algName == "I") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::scalingValue) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("Scaling Value: " + to_string(Conf::scalingValue));

    IntUpperBound intUpperBound(Conf::k, Conf::scalingValue, &q, &p);
    intUpperBound.topK();

  } else if (Conf::algName == "I-SIMD") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::scalingValue) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("Scaling Value: " + to_string(Conf::scalingValue));

    SIMDIntUpperBound simdIntUpperBound(Conf::k, &q, &p);
    simdIntUpperBound.topK();

  } else if (Conf::algName == "S") {
    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::SIGMA) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("SIGMA: " + to_string(Conf::SIGMA));

    SVDIncrPrune svdIncrPrune(Conf::k, Conf::SIGMA, &q, &p);
    svdIncrPrune.topK();

  } else if (Conf::algName == "S-Ind") {
    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::SIGMA) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("SIGMA: " + to_string(Conf::SIGMA));

    SVDIncrPruneIndividualReorder svdIncrPruneIndividualReorder(
        Conf::k, Conf::SIGMA, &q, &p);
    svdIncrPruneIndividualReorder.topK();

  } else if (Conf::algName == "SI") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + "-" +
                         to_string(Conf::scalingValue) + "-" +
                         to_string(Conf::SIGMA) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);
    Logger::Log("SIGMA: " + to_string(Conf::SIGMA));
    Logger::Log("Scaling Value: " + to_string(Conf::scalingValue));

    // Construct index
    SVDIntUpperBoundIncrPrune svdIntUpperBoundIncrPrune(
        Conf::k, Conf::scalingValue, Conf::SIGMA, &q, &p);

#ifdef ONLINE_DECISION_RULE
    std::random_device rd; // only used once to initialise (seed) engine
    std::mt19937 rng(
        rd()); // random-number engine used (Mersenne-Twister in this case)
    unsigned long num_users_per_block =
        4 * (L2_CACHE_SIZE / (sizeof(double) * q.colNum));
    while (num_users_per_block*p.rowNum*sizeof(double) > MAX_MEM_SIZE) {
      num_users_per_block /= 2;
    }
    std::uniform_int_distribution<int> uni(
        0, q.rowNum - num_users_per_block); // guaranteed unbiased
    const unsigned int rand_ind = uni(rng);

    const double blocked_mm_time =
        decisionRuleBlockedMM(q, p, rand_ind, num_users_per_block);

    Monitor tt;
    tt.start();
    svdIntUpperBoundIncrPrune.topK(rand_ind, rand_ind + num_users_per_block);
    tt.stop();

    const double fexipro_time = tt.getElapsedTime();

    Logger::Log("Blocked MM time: " + to_string(blocked_mm_time));
    Logger::Log("FEXIPRO time: " + to_string(fexipro_time));
    if (blocked_mm_time < fexipro_time) {
      Logger::Log("Blocked MM wins");
    } else {
      Logger::Log("FEXIPRO wins");
#ifndef TEST_ONLY
      svdIntUpperBoundIncrPrune.topK(0, rand_ind);
      svdIntUpperBoundIncrPrune.topK(rand_ind + num_users_per_block, q.rowNum);
      svdIntUpperBoundIncrPrune.addToOnlineTime(blocked_mm_time);
      svdIntUpperBoundIncrPrune.outputResults();
#endif
    }
#else
    svdIntUpperBoundIncrPrune.topK(0, q.rowNum);
    svdIntUpperBoundIncrPrune.outputResults();
#endif

  } else if (Conf::algName == "BallTree") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);

    ballTreeTopK(Conf::k, q, p);

  } else if (Conf::algName == "FastMKS") {

    string logFileName = Conf::logPathPrefix + Conf::dataset + "-" +
                         Conf::algName + "-" + to_string(Conf::k) + ".txt";
    Logger::open(logFileName);
    basicLog(q, p, Conf::k);

    fastMKS(Conf::k, Conf::pDataPath, Conf::qDataPath);
  } else {
    cout << "unrecognized method" << endl;
  }

  return 0;
}

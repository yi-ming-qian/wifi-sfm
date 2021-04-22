#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <time.h>
#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

typedef vector<int> VecInt;
typedef vector<VecInt> MatInt;
static constexpr int kStride = 100;

struct Corres{
    int id1;
    int id2;
    double dist;
};

class Ronin{
    public:
        ~Ronin(){
            delete[] mags;
            delete[] thetas;
            delete[] biases;
            delete[] biases_copy;
        }

        double* get_biases() {return biases;}
        MatInt& get_cluster_indices() {return cluster_indices;}
        VecInt& get_cluster_indices(int i) {return cluster_indices[i];}
        double* get_mags() {return mags;}
        double* get_thetas() {return thetas;}
        int get_n_velocity() {return n_velocity;}
        int get_n_corres() {return n_corres;}
        vector<Corres>& get_corres_pairs() {return corres_pairs;}
        VecInt& get_r_corres_idx() {return r_corres_idx;}

        bool write_mag_theta(string &filename){
            for(int i=0;i<n_velocity;i++){
                biases[i] = biases_copy[i];
            }
        	ofstream fout(filename);
        	fout<<n_velocity<<endl;
        	double updated_x=0., updated_y=0.;
        	for (int i=0; i<n_velocity; i++){
        		updated_x += mags[i]*cos(thetas[i]+biases[i]);
        		updated_y += mags[i]*sin(thetas[i]+biases[i]);
        		fout<<mags[i]<<"\t"<<thetas[i]<<"\t"<<biases[i]<<"\t"<<updated_x<<"\t"<<updated_y<<endl;
        	}
        	fout.close();
        	return true;
        }
        void reset_bias(){
            for (int i=0;i<n_velocity;i++){
                biases[i] = 0.;
            }
        }

        bool read_mag_theta(string &filename){
            ifstream fin;
            // read mag and theta
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read mag theta"<<endl;
                exit(1);
            }
            string tmp;
            fin>>tmp>>n_velocity;
            mags = new double[n_velocity];
            thetas = new double[n_velocity];
            biases = new double[n_velocity];
            biases_copy = new double[n_velocity];
            for (int i=0; i<n_velocity; i++){
                fin>>mags[i]>>thetas[i];
                biases[i] = 0.;
                biases_copy[i] = 0.;
            }
            fin.close();
            // for (int i=0; i<30; i++){
            //     cout<<mags[i]<<"\t"<<thetas[i]<<endl;
            // }
            return true;
        }
        bool read_corres_data(string& filename){
            ifstream fin;
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read corres data"<<endl;
                exit(1);
            }
            string tmp;
            fin>>tmp>>n_corres;
            corres_pairs.resize(n_corres);
            for (int i=0; i<n_corres; i++){
                double t1,t2,t3;
                fin>>t1>>t2>>t3;
                //cout<<t1<<" "<<t2<<" "<<t3<<endl;
                corres_pairs[i].id1 = t1;
                corres_pairs[i].id2 = t2;
                corres_pairs[i].dist = t3;
                //cout<<corres_pairs[i].id1<<"\t"<<corres_pairs[i].id2<<"\t"<<corres_pairs[i].dist<<endl;
            }
            fin.close();
            double t_sum = 0.;
            for (int i=0; i<n_corres; i++){
                corres_pairs[i].dist = std::exp(-corres_pairs[i].dist/50.);
                t_sum += corres_pairs[i].dist;
            }
            for (int i=0; i<n_corres; i++){
                corres_pairs[i].dist /= t_sum;
            }

            // cout<<n_corres<<endl;
            // for (int i=0; i<n_corres; i++){
            //     cout<<corres_pairs[i].id1<<"\t"<<corres_pairs[i].id2<<"\t"<<corres_pairs[i].dist<<endl;
            // }
            
            return true;
        }
        bool read_cluster_indices(string &filename){
            ifstream fin;
            // read mag and theta
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read cluster indices"<<endl;
                exit(1);
            }
            int tmp;
            fin>>tmp;
            cluster_indices.resize(tmp);
            for (size_t i=0;i<cluster_indices.size();i++){
                fin>>tmp;
                fin>>tmp;
                cluster_indices[i].resize(tmp);
                for (int j=0;j<tmp;j++){
                    fin>>cluster_indices[i][j];
                }
            }
            fin.close();
            // print
            // for (size_t i=0;i<cluster_indices.size();i++){
            //     for (size_t j=0;j<cluster_indices[i].size();j++){
            //         cout<<cluster_indices[i][j]<<"\t";
            //     }
            //     cout<<endl;
            // }
            return true;
        }
        void sample_corres_pairs(int n_reduce){
            r_corres_idx.resize(n_reduce);
            std::unordered_set<unsigned int> indices;
            while (indices.size() < n_reduce)
                indices.insert(rand()%n_corres);
            int i=0;
            for (auto iter=indices.begin(); iter!=indices.end();iter++){
                r_corres_idx[i]=(*iter);
                i++;
            }
            // for (i=0;i<r_corres_idx.size();i++){
            //     cout<<r_corres_idx[i]<<"\t";
            // }
            // cout<<endl;
        }
        bool comp_inlier_ratio(string& in_path){
            double xstart=0., ystart= 0.;
            vector<double> updated_x(n_velocity+1);
            vector<double> updated_y(n_velocity+1);
            updated_y[0] = ystart;
            updated_x[0] = xstart;
            for (int i=0;i<n_velocity;i++){
                xstart += mags[i]*cos(thetas[i]+biases[i]);
                ystart += mags[i]*sin(thetas[i]+biases[i]);
                updated_x[i+1] = xstart;
                updated_y[i+1] = ystart;
            }
            vector<double> corres_dist_2d(corres_pairs.size());
            double avg_dist_2d = 0.;
            for(size_t i=0;i<corres_pairs.size();i++){
                int id1 = corres_pairs[i].id1;
                int id2 = corres_pairs[i].id2;
                corres_dist_2d[i] = fabs(updated_x[id1] - updated_x[id2])+fabs(updated_y[id1] - updated_y[id2]);
                avg_dist_2d += corres_dist_2d[i];
            }
            avg_dist_2d /= corres_pairs.size();
            // sort(corres_dist_2d.begin(), corres_dist_2d.end());
            // double med_dist_2d = corres_dist_2d[int(corres_dist_2d.size()/2.)];
            double inlier_ratio = 0.;
            VecInt r_corres_idx_copy(r_corres_idx);
            r_corres_idx.clear();
            for(size_t i=0;i<corres_pairs.size();i++){
                if (corres_dist_2d[i]<avg_dist_2d){
                    r_corres_idx.push_back(i);
                    inlier_ratio += 1.;
                }
            }
            cout<<r_corres_idx.size()<<"\t"<<corres_pairs.size()<<endl;
            string filename = in_path + "c_single_corres_sample_incremental.txt";
            ofstream fout(filename);
            for (size_t i=0;i<r_corres_idx.size();i++){
                fout<<r_corres_idx[i]<<"\t";
            }
            fout<<endl;
            fout.close();
            if (r_corres_idx_copy==r_corres_idx)
                return true;
            return false;
        }
        void copy_bias(){
            for(int i=0;i<n_velocity;i++){
                biases_copy[i] = biases[i];
            }
        }

    private:
        int n_velocity;
        int n_corres;
        double* mags;
        double* thetas;
        MatInt cluster_indices;
        double* biases;
        double* biases_copy;
        vector<Corres> corres_pairs;
        VecInt r_corres_idx; //sampled reduced corres pair idx

};

struct StatConstraint {
    typedef DynamicAutoDiffCostFunction<StatConstraint, kStride>
        StatCostFunction;

    StatConstraint(Ronin* ronin_ptr)
        : ronin_ptr(ronin_ptr) {}

    template <typename T>
    bool operator()(T const* const* biases, T* residuals) const {
    	double* mags = ronin_ptr->get_mags();
    	double* thetas = ronin_ptr->get_thetas();
        int n_velocity = ronin_ptr->get_n_velocity();

        T xstart=T(0.), ystart= T(0.);
        vector<T> updated_x(n_velocity+1);
        vector<T> updated_y(n_velocity+1);
        updated_y[0] = ystart;
        updated_x[0] = xstart;
        for (int i=0;i<n_velocity;i++){
            xstart += mags[i]*cos(thetas[i]+biases[i][0]);
            ystart += mags[i]*sin(thetas[i]+biases[i][0]);
            updated_x[i+1] = xstart;
            updated_y[i+1] = ystart;
        }
        vector<Corres>& corres_pairs = ronin_ptr->get_corres_pairs();
        VecInt& r_corres_idx = ronin_ptr->get_r_corres_idx();
        size_t pair_size = r_corres_idx.size();
        for (size_t k=0; k<pair_size; k++) {
            int i = r_corres_idx[k];
            int id1 = corres_pairs[i].id1;
            int id2 = corres_pairs[i].id2;
            double dist = corres_pairs[i].dist;
            residuals[k] = (updated_x[id1] - updated_x[id2]);//*dist;
            residuals[k+pair_size] = (updated_y[id1] - updated_y[id2]);//*dist;
        }
    	return true;
    }
    static StatCostFunction* Create(Ronin* ronin_ptr,
    	                            vector<double*>* parameter_blocks) {
    	StatConstraint* constraint = new StatConstraint(ronin_ptr);
    	StatCostFunction* cost_function = new StatCostFunction(constraint);
    	// delete this new memory?
    	parameter_blocks->clear();
    	double* biases = ronin_ptr->get_biases();
        int n_velocity = ronin_ptr->get_n_velocity();
    	for (int i=0; i<n_velocity; i++) {
    		parameter_blocks->push_back(&(biases[i]));
    		cost_function->AddParameterBlock(1);
    	}
        size_t n_corres = ronin_ptr->get_r_corres_idx().size();

    	cost_function->SetNumResiduals(n_corres*2);
    	return (cost_function);
    }
    ///////
    int cluster_id;
    Ronin* ronin_ptr;
};

struct BiasRegConstraint {
	template <typename T>
	bool operator()(const T* const b1, const T* const b2, T* residual) const {
		//residual[0] = min(ceres::abs(b1[0]-b2[0]), T(2.*M_PI)-ceres::abs(b1[0]-b2[0]));
        residual[0] = b1[0]-b2[0];
		return true;
	}
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    srand(time(NULL));
    string folder_id = argv[1];
    string in_path = "/local-scratch/yimingq/wifi/stationary/outputs/"+folder_id+"/";
    Ronin ronin;
    string txtname = in_path + "c_mag_theta.txt";
    ronin.read_mag_theta(txtname);
    txtname = in_path + "c_single_corres.txt";
    ronin.read_corres_data(txtname);
    cout<<"finish reading"<<endl;

    //MatInt& cluster_indices = ronin.get_cluster_indices();
    double* biases = ronin.get_biases();
    int n_velocity = ronin.get_n_velocity();
    double* thetas = ronin.get_thetas();
    int n_corres = ronin.get_n_corres();

    // do ransac
    if (n_corres>0){
        int iter_num = 10;
        int n_chosen = min(0.15*n_corres, 30.);
        if (n_chosen<=1){
            n_chosen=n_corres;
            iter_num=1;
        }
        ronin.sample_corres_pairs(n_chosen);

        double best_inratio = 0.;
        for (int r=0;r<iter_num;r++){
            cout<<"--------iteration: "<<r<<"--------"<<endl;
            ronin.reset_bias();
            // ronin.sample_corres_pairs(n_chosen);
            Problem problem;

            vector<double*> parameter_blocks;
            StatConstraint::StatCostFunction* stat_cost_function = 
            	StatConstraint::Create(&ronin, &parameter_blocks);
            problem.AddResidualBlock(stat_cost_function, new CauchyLoss(0.5), parameter_blocks);
        	
        	for (int i=1; i<n_velocity; i++){
        		problem.AddResidualBlock(
        			new AutoDiffCostFunction<BiasRegConstraint, 1, 1, 1>(new BiasRegConstraint),
        			new CauchyLoss(0.5),
        			&(biases[i]),
        			&(biases[i-1]));
        	}
            // for (int i=0; i<n_velocity; i++){
            // 	problem.SetParameterLowerBound(&(biases[i]), 0, -M_PI-thetas[i]);
            // 	problem.SetParameterUpperBound(&(biases[i]), 0, M_PI-thetas[i]);
            // }
    	
    		ceres::Solver::Options options;
    		//options.linear_solver_type = ceres::DENSE_QR;
            //options.max_num_iterations = 30;
    		options.minimizer_progress_to_stdout = true;
    		ceres::Solver::Summary summary;
    		ceres::Solve(options, &problem, &summary);
    		cout << summary.FullReport() << "\n";

            // check whetehr the best
            //double inlier_ratio = ronin.comp_inlier_ratio();
            if (ronin.comp_inlier_ratio(in_path)){
                //best_inratio=inlier_ratio;
                ronin.copy_bias();
                break;
            }
    	}
    }
	txtname = in_path + "c_single_corres_align_incremental.txt";
	ronin.write_mag_theta(txtname);

    cout<<"success!"<<endl;
    return 0;
}

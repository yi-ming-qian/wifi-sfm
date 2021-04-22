#include <iostream>
#include <fstream>
#include <sstream>
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

class Ronin{
    public:
        ~Ronin(){
            delete[] mags;
            delete[] thetas;
            delete[] biases;
        }

        double* get_biases() {return biases;}
        MatInt& get_cluster_indices() {return cluster_indices;}
        VecInt& get_cluster_indices(int i) {return cluster_indices[i];}
        double* get_mags() {return mags;}
        double* get_thetas() {return thetas;}
        int get_n_velocity() {return n_velocity;}

        bool write_mag_theta(string &filename){
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
            for (int i=0; i<n_velocity; i++){
                fin>>mags[i]>>thetas[i];
                biases[i] = 0.;
            }
            fin.close();
            // for (int i=0; i<30; i++){
            //     cout<<mags[i]<<"\t"<<thetas[i]<<endl;
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

    private:
        int n_velocity;
        double* mags;
        double* thetas;
        MatInt cluster_indices;
        double* biases;

};

struct StatConstraint {
    typedef DynamicAutoDiffCostFunction<StatConstraint, kStride>
        StatCostFunction;

    StatConstraint(int cluster_id,
                   Ronin* ronin_ptr)
        : cluster_id(cluster_id),
          ronin_ptr(ronin_ptr) {}

    template <typename T>
    bool operator()(T const* const* biases, T* residuals) const {
    	VecInt& cluster_indices = ronin_ptr->get_cluster_indices(cluster_id);
    	size_t cluster_size = cluster_indices.size();
    	double* mags = ronin_ptr->get_mags();
    	double* thetas = ronin_ptr->get_thetas();

    	vector<T> updated_x(cluster_size);
    	vector<T> updated_y(cluster_size);
    	T xstart=T(0.), ystart= T(0.);
    	for (size_t i=0; i<cluster_size; i++){
    		int traj_id = cluster_indices[i];
    		int traj_id_prev = 0;
    		if (i!=0)
    			traj_id_prev = cluster_indices[i-1];
    		for (int j=traj_id_prev; j<traj_id; j++){
    			xstart += mags[j]*cos(thetas[j]+biases[j][0]);
    			ystart += mags[j]*sin(thetas[j]+biases[j][0]);
    		}
    		updated_x[i] = xstart;
    		updated_y[i] = ystart;
    	}
    	T xmean=T(0.), ymean=T(0.);
    	for (size_t i=0; i<cluster_size; i++){
    		xmean += updated_x[i];
    		ymean += updated_y[i];
    	}
        xmean /= T(cluster_size);
        ymean /= T(cluster_size);
    	for (size_t i=0; i<cluster_size; i++){
    		residuals[i] = updated_x[i] - xmean;
    		residuals[i+cluster_size] = updated_y[i] - ymean;
    	}
    	return true;
    }
    static StatCostFunction* Create(int cluster_id,
    	                            Ronin* ronin_ptr,
    	                            vector<double*>* parameter_blocks) {
    	StatConstraint* constraint = new StatConstraint(cluster_id, ronin_ptr);
    	StatCostFunction* cost_function = new StatCostFunction(constraint);
    	// delete this new memory?
    	parameter_blocks->clear();
    	double* biases = ronin_ptr->get_biases();
    	VecInt& cluster_index = ronin_ptr->get_cluster_indices(cluster_id);
    	int last_traj_id = cluster_index[cluster_index.size()-1];
    	for (int i=0; i<last_traj_id; i++) {
    		parameter_blocks->push_back(&(biases[i]));
    		cost_function->AddParameterBlock(1);
    	}
    	cost_function->SetNumResiduals(cluster_index.size()*2);
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

    string folder_id = argv[1];
    string in_path = "/cs/vml-furukawa/user/yimingq/wifi/stationary/outputs/"+folder_id+"/";
    Ronin ronin;
    string txtname = in_path + "c_mag_theta.txt";
    ronin.read_mag_theta(txtname);
    txtname = in_path + "c_cluster_indices.txt";
    ronin.read_cluster_indices(txtname);
    cout<<"finish reading"<<endl;

    MatInt& cluster_indices = ronin.get_cluster_indices();
    double* biases = ronin.get_biases();
    int n_velocity = ronin.get_n_velocity();
    double* thetas = ronin.get_thetas();

    Problem problem;
    bool do_optimize = false;
    for (size_t i=0; i<cluster_indices.size(); i++){
        if (cluster_indices[i].size()==1)
            continue;
        do_optimize = true;
        vector<double*> parameter_blocks;
        StatConstraint::StatCostFunction* stat_cost_function = 
        	StatConstraint::Create(
        		i, &ronin, &parameter_blocks);
        problem.AddResidualBlock(stat_cost_function, new CauchyLoss(0.5), parameter_blocks);
    }
    
	
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

	
	if (do_optimize) {
		ceres::Solver::Options options;
		//options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		cout << summary.FullReport() << "\n";
	}

	txtname = in_path + "c_singlealign.txt";
	ronin.write_mag_theta(txtname);

    cout<<"success!"<<endl;
    return 0;
}

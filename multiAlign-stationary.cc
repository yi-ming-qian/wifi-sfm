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

struct Cluster{
    int label;
    VecInt indices;
};

class Ronin{
    public:
        ~Ronin(){
            delete[] mags;
            delete[] thetas;
            //delete[] biases;
        }

        //double* get_biases() {return biases;}
        vector<Cluster>& get_cluster_infos() {return cluster_infos;}
        //VecInt& get_cluster_indices(int i) {return cluster_indices[i];}
        double* get_mags() {return mags;}
        double* get_thetas() {return thetas;}
        int get_n_velocity() {return n_velocity;}
        double* get_gtransforms() {return gtransfroms;}

        bool write_mag_theta(string &filename){
        	ofstream fout(filename);
        	fout<<n_velocity<<endl;
        	double updated_x=gtransfroms[0], updated_y=gtransfroms[1];
        	for (int i=0; i<n_velocity; i++){
        		updated_x += mags[i]*cos(thetas[i]+gtransfroms[2]);
        		updated_y += mags[i]*sin(thetas[i]+gtransfroms[2]);
        		fout<<mags[i]<<"\t"<<thetas[i]<<"\t"<<gtransfroms[2]<<"\t"<<updated_x<<"\t"<<updated_y<<endl;
        	}
            fout<<gtransfroms[0]<<"\t"<<gtransfroms[1]<<endl;
        	fout.close();
        	return true;
        }

        bool read_mag_theta(string &filename){//read single align result
            ifstream fin;
            // read mag and theta
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read mag theta"<<endl;
                exit(1);
            }
            string tmp;
            fin>>n_velocity;
            mags = new double[n_velocity];
            thetas = new double[n_velocity];
            double bias;
            for (int i=0; i<n_velocity; i++){
                fin>>mags[i]>>thetas[i]>>bias>>tmp>>tmp;
                thetas[i] += bias;
            }
            fin.close();
            // for (int i=0; i<30; i++){
            //     cout<<mags[i]<<"\t"<<thetas[i]<<endl;
            // }
            return true;
        }
        bool read_cluster_infos(string &filename){
            ifstream fin;
            // read mag and theta
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read cluster indices"<<endl;
                exit(1);
            }
            int tmp;
            fin>>tmp;
            cluster_infos.resize(tmp);
            for (size_t i=0;i<cluster_infos.size();i++){
                fin>>cluster_infos[i].label>>tmp;
                cluster_infos[i].indices.resize(tmp);
                for (int j=0;j<tmp;j++){
                    fin>>cluster_infos[i].indices[j];
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
        vector<Cluster> cluster_infos;
        //double* biases;
        double gtransfroms[3] = {0.}; //global transform (startx, starty, global biase)

};

struct StatConstraint {
    typedef DynamicAutoDiffCostFunction<StatConstraint, kStride>
        StatCostFunction;

    StatConstraint(int cluster_id,
                   vector<Ronin>* all_ronins)
        : cluster_id(cluster_id),
          all_ronins(all_ronins) {}

    template <typename T>
    bool operator()(T const* const* params, T* residuals) const {
        int pid = 0; // paramater id
        vector<T> updated_x;
        updated_x.reserve(100);
    	vector<T> updated_y;
        updated_y.reserve(100);
        for (size_t d=0; d<(*all_ronins).size(); d++){
            vector<Cluster>& cluster_infos = (*all_ronins)[d].get_cluster_infos();
            double* mags = (*all_ronins)[d].get_mags();
    	    double* thetas = (*all_ronins)[d].get_thetas();
            for (size_t i=0; i<cluster_infos.size(); i++){
                if (cluster_infos[i].label==cluster_id){
                    T xstart= T(params[pid][0]), ystart = T(params[pid+1][0]);
                    T gbias = T(params[pid+2][0]);
                    pid += 3;
                    size_t numindices = cluster_infos[i].indices.size();
                    for (size_t j=0; j<numindices; j++){
                        int traj_id = cluster_infos[i].indices[j];
                        int traj_id_prev = 0;
                        if (j!=0)
                            traj_id_prev = cluster_infos[i].indices[j-1];
                        for (int k=traj_id_prev; k<traj_id; k++){
                            xstart += mags[k]*cos(thetas[k]+gbias);
                            ystart += mags[k]*sin(thetas[k]+gbias);
                        }
                        updated_x.push_back(xstart);
                        updated_y.push_back(ystart);
                    }
                    break;
                }
            }
        }
        size_t total_nindices = updated_x.size();
        if (cluster_id==0){ // the largest cluster
            for (size_t i=0; i<total_nindices; i++){
                residuals[i] = updated_x[i];
                residuals[i+total_nindices] = updated_y[i];
            }
        }else{
            T xmean=T(0.), ymean=T(0.);
            for (size_t i=0; i<total_nindices; i++){
                xmean += updated_x[i];
                ymean += updated_y[i];
            }
            xmean /= T(total_nindices);
            ymean /= T(total_nindices);
            for (size_t i=0; i<total_nindices; i++){
                residuals[i] = updated_x[i] - xmean;
                residuals[i+total_nindices] = updated_y[i] - ymean;
            }
        }
    	
    	return true;
    }
    static StatCostFunction* Create(int cluster_id,
    	                            vector<Ronin>* all_ronins,
    	                            vector<double*>* parameter_blocks) {
    	StatConstraint* constraint = new StatConstraint(cluster_id, all_ronins);
    	StatCostFunction* cost_function = new StatCostFunction(constraint);
    	// delete this new memory?
    	parameter_blocks->clear();
        size_t nindices = 0;
        for (size_t d=0; d<(*all_ronins).size(); d++){
            // Day d
            vector<Cluster>& cluster_infos = (*all_ronins)[d].get_cluster_infos();
            double* gtransfroms = (*all_ronins)[d].get_gtransforms();
            for (size_t i=0; i<cluster_infos.size(); i++){
                if (cluster_infos[i].label==cluster_id){
                    nindices += cluster_infos[i].indices.size();
                    for(int t=0;t<3;t++){
                        parameter_blocks->push_back(&(gtransfroms[t]));
                        cost_function->AddParameterBlock(1);
                    }
                    break;
                }
            }

        }
    	cost_function->SetNumResiduals(nindices*2);
    	return (cost_function);

    }
    ///////
    int cluster_id;
    vector<Ronin>* all_ronins;
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

    string base_path = "./outputs/";
    ifstream fin(base_path+"folder_list.txt");
    vector<string> folder_ids;
    while(!fin.eof()){
        string tmp;
        fin>>tmp;
        if (tmp==""||tmp=="\n")
            continue;
        folder_ids.push_back(tmp);
    }
    fin.close();
    vector<Ronin> all_ronins;
    all_ronins.resize(folder_ids.size());
    for (size_t i=0;i<folder_ids.size();i++){
        string txtname = base_path+folder_ids[i] + "/c_singlealign.txt";
        cout<<txtname<<endl;
        all_ronins[i].read_mag_theta(txtname);
        txtname = base_path+folder_ids[i] + "/c_cluster_indices.txt";
        all_ronins[i].read_cluster_infos(txtname);
    }
    
    cout<<"finish reading"<<endl;
    Problem problem;
    for (size_t i=0; i<10; i++){ // you have 10 clusters
        vector<double*> parameter_blocks;
        StatConstraint::StatCostFunction* stat_cost_function = 
        	StatConstraint::Create(
        		i, &all_ronins, &parameter_blocks);
        problem.AddResidualBlock(stat_cost_function, new CauchyLoss(0.5), parameter_blocks);
        //problem.AddResidualBlock(stat_cost_function, NULL, parameter_blocks);
    }

    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << "\n";

    for (size_t i=0;i<folder_ids.size();i++){
        string txtname = base_path+folder_ids[i] + "/c_multialign.txt";
        all_ronins[i].write_mag_theta(txtname);
    }

    cout<<"success!"<<endl;
    return 0;
}

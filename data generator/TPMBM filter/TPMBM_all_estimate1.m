function [X_estimate,X_cov_estimate,X_length_estimate,t_b_estimate,length_estimate,existence_estimate,alive_prob]=TPMBM_all_estimate1(filter_upd,existence_estimation_threshold,Nx,k)

%Author: Angel F. Garcia Fernandez

%Option 1: one picks global hypothesis with highest weight and then takes the
%trajectory Bernoullis  above a certain threshold

%Then, for each Bernoulli, we estimate the trajectory end time with highest probability of existence

X_estimate=cell(0,1);
X_cov_estimate=cell(0,1);
X_length_estimate=cell(0,1);
t_b_estimate=[];
length_estimate=[];
existence_estimate=[];
alive_prob=[];

if(~isempty(filter_upd.globHyp))
    globHypWeight=filter_upd.globHypWeight;
    [~,index]=max(globHypWeight);
    
    HypMax=filter_upd.globHyp(index,:);
    
    index_output=1;
    
    for i=1:length(HypMax)
        hyp_i=HypMax(i);
        if(hyp_i>0)
            Existence=filter_upd.tracks{i}.eB(hyp_i);
            if(Existence>existence_estimation_threshold)
                
                t_b_estimate=[t_b_estimate,filter_upd.tracks{i}.t_b];
                existence_estimate=[existence_estimate Existence];
                
                prob_length_j=filter_upd.tracks{i}.prob_length{hyp_i};
                [~,index_prob]=max(prob_length_j);
                X_length_estimate{index_output}=flip(prob_length_j);
                if(index_prob==1)
                    %Trajectory is alive
                    X_estimate{index_output}=filter_upd.tracks{i}.meanB{hyp_i};
                    X_cov_estimate{index_output}=filter_upd.tracks{i}.covB{hyp_i};
                    length_estimate=[length_estimate,filter_upd.tracks{i}.length];
                    alive_prob = [alive_prob,prob_length_j(1)];
                else
                    X_estimate{index_output}=filter_upd.tracks{i}.mean_past{hyp_i}{index_prob-1};
                    X_cov_estimate{index_output}=filter_upd.tracks{i}.cov_past{hyp_i}{index_prob-1};
                    length_estimate=[length_estimate,length(X_estimate{index_output})/Nx]; 
                    if filter_upd.tracks{i}.t_b + length(prob_length_j) - 1 < k
                        alive_prob = [alive_prob,0];
                    else
                        alive_prob = [alive_prob,prob_length_j(1)];
                    end
                end
         
                index_output=index_output+1;
            end
        end
    end
end
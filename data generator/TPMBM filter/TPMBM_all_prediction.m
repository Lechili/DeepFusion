function filter_pred=TPMBM_all_prediction(filter_upd,F,Q,p_s,weights_b,means_b,covs_b,Lscan,k,T_alive)
%Author: Angel F. Garcia-Fernandez

%Prediction for Poisson component (we only keep track of alive trajectories
%in the Poisson component)

Ncom=length(filter_upd.Pois);
Nx=size(F,1);

for i=1:Ncom
    filter_pred.Pois{i}.weightPois=p_s*filter_upd.Pois{i}.weightPois;
    filter_pred.Pois{i}.meanPois=[filter_upd.Pois{i}.meanPois;F*filter_upd.Pois{i}.meanPois(end-Nx+1:end)];
    
    %Prediction of the covariance matrix
    length_k_i=filter_upd.Pois{i}.length_Pois;
    if(Lscan>1)
        min_length=min([Lscan-1,length_k_i]);
        cov_i=filter_upd.Pois{i}.covPois(end-Nx*min_length+1:end,end-Nx*min_length+1:end);
        cov_i_pred=F*cov_i(end-Nx+1:end,end-Nx+1:end)*F'+Q;
        F_mode=zeros(Nx,size(cov_i,1));
        F_mode(end-3:end,end-3:end)=F;
        cross=F_mode*cov_i;
        cov_upd_i=[cov_i,cross';cross,cov_i_pred];
    else
        cov_i=filter_upd.Pois{i}.covPois;
        cov_upd_i=F*cov_i*F'+Q;
        
    end
    
    filter_pred.Pois{i}.covPois=cov_upd_i;
    %Now we predict the start time and length of the trajectory
    filter_pred.Pois{i}.t_bPois=filter_upd.Pois{i}.t_bPois;
    filter_pred.Pois{i}.length_Pois=filter_upd.Pois{i}.length_Pois+1;
end
%We add the PHD of new born trajectories
for i=1:length(weights_b)
    filter_pred.Pois{Ncom+i}.weightPois=weights_b(i);
    filter_pred.Pois{Ncom+i}.meanPois=means_b(:,i);
    filter_pred.Pois{Ncom+i}.covPois=squeeze(covs_b(:,:,i));
    filter_pred.Pois{Ncom+i}.t_bPois=k+1; %This birth component is born at time k+1
    filter_pred.Pois{Ncom+i}.length_Pois=1;
end




%Prediction for Bernoulli components
filter_pred.globHyp=filter_upd.globHyp;
filter_pred.globHypWeight=filter_upd.globHypWeight;


Ntracks=length(filter_upd.tracks);

if(Ntracks>0)
    for i=1:Ntracks
        Nhyp_i=length(filter_upd.tracks{i}.eB);
        filter_pred.tracks{i}.t_ini=filter_upd.tracks{i}.t_ini;
        %We predict the birth time and length of the trajectory
        filter_pred.tracks{i}.t_b=filter_upd.tracks{i}.t_b;
        filter_pred.tracks{i}.length=filter_upd.tracks{i}.length+1;
        
    
        
        for j=1:Nhyp_i %We go through all hypotheses
            
            
            %We first check if the trajectory is alive (can be alive)
            prob_length_j=filter_upd.tracks{i}.prob_length{j};
            
            if(prob_length_j(1)>T_alive)
                
               prob_alive=p_s*prob_length_j(1);

                %Prediction of the mean
                filter_pred.tracks{i}.meanB{j}=[filter_upd.tracks{i}.meanB{j};F*filter_upd.tracks{i}.meanB{j}(end-Nx+1:end)];
                
                %Prediction of the covariance matrix
                length_k_i=filter_upd.tracks{i}.length;
                
                if(Lscan>1)
                    min_length=min([Lscan-1,length_k_i]);
                    cov_i=filter_upd.tracks{i}.covB{j}(end-Nx*min_length+1:end,end-Nx*min_length+1:end);
                    cov_i_pred=F*cov_i(end-Nx+1:end,end-Nx+1:end)*F'+Q;
                    F_mode=zeros(Nx,size(cov_i,1));
                    F_mode(end-3:end,end-3:end)=F;
                    cross=F_mode*cov_i;
                    cov_upd_i=[cov_i,cross';cross,cov_i_pred];
                else
                    cov_i=filter_upd.tracks{i}.covB{j};
                    cov_upd_i=F*cov_i*F'+Q;
                end
                
                
                filter_pred.tracks{i}.covB{j}=cov_upd_i;
                filter_pred.tracks{i}.eB(j)=filter_upd.tracks{i}.eB(j); %Difference w.r.t. alive trajectories
                filter_pred.tracks{i}.aHis{j}=filter_upd.tracks{i}.aHis{j};
                
                %We update the probability of length
                filter_pred.tracks{i}.prob_length{j}=[prob_alive;(1-p_s)*prob_length_j(1);filter_upd.tracks{i}.prob_length{j}(2:end)];
                    
                
                %We update the past means of the trajectories
                filter_pred.tracks{i}.mean_past{j}{1}=filter_upd.tracks{i}.meanB{j};
                filter_pred.tracks{i}.cov_past{j}{1}=filter_upd.tracks{i}.covB{j};
                %Npast_means=length(prob_length_j)-1;
                Npast_means=length(filter_upd.tracks{i}.mean_past{j});
                for p=1:Npast_means
                    filter_pred.tracks{i}.mean_past{j}{p+1}=filter_upd.tracks{i}.mean_past{j}{p};
                    filter_pred.tracks{i}.cov_past{j}{p+1}=filter_upd.tracks{i}.cov_past{j}{p};
                end
                    
                
            else
               %We just copy paste the previous Bernoulli component

               filter_pred.tracks{i}.meanB{j}=filter_upd.tracks{i}.meanB{j};
               filter_pred.tracks{i}.covB{j}=filter_upd.tracks{i}.covB{j};
               filter_pred.tracks{i}.eB(j)=filter_upd.tracks{i}.eB(j);
               filter_pred.tracks{i}.aHis{j}=filter_upd.tracks{i}.aHis{j};
               filter_pred.tracks{i}.prob_length{j}=prob_length_j;
               filter_pred.tracks{i}.mean_past{j}=filter_upd.tracks{i}.mean_past{j};
               filter_pred.tracks{i}.cov_past{j}=filter_upd.tracks{i}.cov_past{j};
            end
                        
            
        end
        
    end
else
    filter_pred.tracks=cell(0,1);
    filter_pred.globHyp=[];
    filter_pred.globHypWeight=[];
end




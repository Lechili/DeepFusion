clear
clc

addpath eval/

losses = load('eval_loss_mt3v2_S1_200k_standard_lrp_50k_rn_5_all_2022-05-13_112313');
%losses = struct2cell(losses);

n_steps = length(losses.training_losses);
figure(1)
grid on; 
semilogx([1:n_steps],losses.training_losses,'r'); 
hold on
plot(losses.eval_step,losses.eval_losses,'b')
title('Training and validation losses')
xlabel('Gradient Steps')
ylabel('Last layer Losses')
legend('Training Loss','Evaluation Losses')

figure(2)
grid on; 
semilogx([1:n_steps],losses.training_total_losses,'r'); 
hold on
semilogx(losses.eval_step,losses.eval_total_losses,'b')
title('Training and validation losses',FontSize=15)
xlabel('Gradient Steps',FontSize=15)
ylabel('Total Losses',FontSize=15)
xlim([0,420000])
legend('Training Loss','Evaluation Losses',FontSize=15)
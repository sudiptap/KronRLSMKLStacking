function [alpha, best_kernel] = mkboost_d1_mplusn(y_train, K, K_comb, dist, k1_k2, lambda)
	%parameters
        T = 10;
	%n = 
	alpha = zeros(T);
	best_kernel = zeros(T);
	%K1
	if k1_k2
		K1 = K;
		K2 = K_comb;
	else
		K1 = K_comb;
		
	end	

        for t=1:T
		%call the sampling here
		%[] = sample_example();
		loss = realmax('single');
		best_kernel_index = -1;
disp(size(K,3));
		for j=1:size(K,3)
			%train a week classifier/predictor with kernel k_j
			if k1_k2			
				[A] = kronrls(K(:,:,j), K_comb, y_train, lambda);
			else
				[A] = kronrls(K_comb, K(:,:,j), y_train, lambda);
			end
			%evaluate training performance on train_idx only i.e. A(train_idx) and y_train(train_idx)
			%get the best kernel based on loss incurred
			[current_loss, predicted] = getLoss_1(A, y_train, dist);
			if loss> current_loss
				loss = current_loss;
				best_kernel_index = j;				
			end
		end
		%loss is minimum of all losses
		alpha(t) = 0.5 * log((1-loss)/(loss));		
		best_kernel(t) = best_kernel_index;
		old_dist = dist(:,:);
		for dist_i=1:size(dist,1)
			for dist_j=1:size(dist,2)
				if y_train(dist_i, dist_j) == 1		
					if A(dist_i, dist_j)>=(y_train(dist_i, dist_j)*0.5)		
						dist(dist_i, dist_j)=old_dist(dist_i, dist_j) * exp((-alpha(t))*t);
					else
						dist(dist_i, dist_j)=old_dist(dist_i, dist_j) * exp((alpha(t))*t);
					end
				end
			end
		end
		sum_dist = sum(sum(dist));
		dist = dist ./ sum_dist;
	end
end

function [] = sample_examples()
	
end

%computes absolute loss
function [loss, predicted] = getLoss_1(predicted, original, dist)
	%predicted = predicted .* dist;
	for i=1:size(predicted,1)
		for j=1:size(predicted,2)
			if predicted(i,j)>0.08
				predicted(i,j)=1;
			else
				predicted(i,j)=0;
			end
		end
	end
	loss_matrix = abs(original-predicted); %.* dist;
	loss = 0;
	for i=1:size(predicted,1)
		for j=1:size(predicted,2)
			if predicted(i,j)~=original(i,j)
				loss = loss + dist(i,j);
			end	
		end
	end
	
	%loss_matrix = loss_matrix ./ original;
	%end
	%loss = sum(sum(loss_matrix));	%disp(loss);
	%loss = sum(sum(loss_matrix))/ num_ones; %disp(loss); return;
	%loss = sum(sum(loss_matrix)); 
	%disp(sum(sum(loss_matrix))); return;
end

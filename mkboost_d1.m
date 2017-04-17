function [alpha, best_kernel] = mkboost_d1(y_train, kernels, dist, lambda, num_ones)
	%disp(size(kernels,2)); return;
	%parameters
        T = 50;
	%n = 
	alpha = zeros(1,T); 
	best_kernel = zeros(1,T);
        for t=1:T
		%disp(t);
		%call the sampling here
		%[] = sample_example();
		loss = realmax('single');
		best_kernel_index = -1;

		for j=1:size(kernels,2)
			%disp(j);
			%train a week classifier/predictor with kernel k_j
			temp = kernels{1,j};
			%disp(size(temp));
			[A] = kronrls(double(temp{1}), double(temp{2}), y_train, lambda);
			%disp(temp{1,2}(1,10));
			%evaluate training performance on train_idx only i.e. A(train_idx) and y_train(train_idx)
			%get the best kernel based on loss incurred
			
			[current_loss, predicted] = getLoss_1(A, y_train, num_ones, dist);
			%disp(current_loss);
			if loss>= current_loss
				loss = current_loss;
				best_kernel_index = j;				
			end
		end
		%loss is minimum of all losses
		%disp(loss);
		alpha(t) = 0.5 * log((1-loss)/(loss)); 
		%disp(alpha(t)); return; 
		best_kernel(t) = best_kernel_index;
		old_dist = dist(:,:);
		for dist_i=1:size(dist,1)
			for dist_j=1:size(dist,2)
				%if y_train(dist_i, dist_j) == 1		
					if predicted(dist_i, dist_j)==(y_train(dist_i, dist_j))		
						%disp('if');
						dist(dist_i, dist_j)=old_dist(dist_i, dist_j) * exp((-alpha(t)));
					else
						%disp('else');
						dist(dist_i, dist_j)=old_dist(dist_i, dist_j) * exp((alpha(t)));
					end
				%end
			end
		end
		sum_dist = sum(sum(dist));
		dist = dist ./ sum_dist;
		%disp(dist);
	end
	%disp(alpha);
	%disp(best_kernel);
	%return;
end

function [] = sample_examples()
	
end

%computes absolute loss
function [loss, predicted] = getLoss_1(predicted, original, num_ones, dist)
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

function [O] = mkboost_d2(y_train_orig, kernels, dist, lambda, num_ones)
	%disp(size(kernels,2)); return;
	%parameters
    T = 50;
	choose_best_classifier = 1; 
	alpha = zeros(T,size(kernels,2));
	alpha1 = zeros(1,T); 
	h = cell(1,T);
	for t=1:T
		h{t}=zeros(size(y_train_orig));
	end
	
    for t=1:T
		%disp(t);
		%call the sampling here		
		%[y_train] = sample_examples(dist, y_train_orig, floor(0.50*size(y_train_orig,1)*size(y_train_orig,2)));
		[y_train] = sample_examples(dist, y_train_orig, floor(0.7*sum(sum(y_train_orig==1))));
		loss = realmax('single');
		best_kernel_index = -1;
		%h{t} = zeros(
		losses = [];
		alpha = [];
		predicted = cell(1,size(kernels,2));
		for j=1:size(kernels,2)
			%disp(j);
			%train a week classifier/predictor with kernel k_j
			temp = kernels{1,j};
			%disp(size(temp));
			[A] = kronrls(double(temp{1}), double(temp{2}), y_train, lambda); 		
			[current_loss, p] = getLoss_1(A, y_train, num_ones, dist);
			if(current_loss>1)
				%disp('hi');
				%disp(current_loss); %return;
			end
			predicted{j} = p;
			losses = [losses current_loss]; 
			%if (current_loss==1)
			%	current_loss = current_loss - 0.0001;
			%	disp(current_loss);
			%end
			%if (current_loss==0)
			%	current_loss = current_loss + 0.0001;
			%	disp(current_loss);
			%end
			%if current_loss==0
				%alpha(t,j)=1;
			%else
				alpha(t,j) = 0.5 * log((1-current_loss)/(current_loss)); 
			%end
			if(~isreal(alpha1(t)))		
				disp(current_loss); %return;
			end			
		end		
		
		if choose_best_classifier
			%disp(losses(2)); 
			%disp(size(predicted));
			[min_loss, min_loss_index] = min(losses);
			%min_loss_index = 10;
			%disp(min_loss_index);
			%if min_loss==0
			%	alpha1(t) = 1;	
			%else
				alpha1(t) = 0.5 * log((1-min_loss)/(min_loss));
				%disp(losses);
			%end
			%disp(predicted{min_loss_index});
			h{t} = h{t} + alpha1(t) .* predicted{min_loss_index};
			%disp(alpha(t,min_loss_index));
			%disp(predicted{min_loss_index});
		else
			%combine the classifiers
			h{t} = alpha(t,1) .* predicted{1};
			for j=2:size(kernels,2)
				h{t} = h{t} + alpha(t,j) .* predicted{j};
			end		
			[current_loss] = getLoss_1(h{t}, y_train, num_ones, dist); 	
						
			alpha1(t) = 1 * log((1-current_loss)/(current_loss));	
						
		end
		%disp(alpha1);
		%if(~isreal(alpha1(t)))		
		%	disp(current_loss); return;
		%end
		%disp(alpha1);
		old_dist = dist(:,:);
		for dist_i=1:size(dist,1)
			for dist_j=1:size(dist,2)
				if y_train(dist_i, dist_j) == 1		
					%if h{t}(dist_i, dist_j)==(y_train(dist_i, dist_j)) && y_train(dist_i, dist_j)==1
						%disp('if');
						dist(dist_i, dist_j)=old_dist(dist_i, dist_j) * exp((-alpha1(t)));
						%if isnan(dist(dist_i, dist_j))
						%	disp(alpha1(t));
						%end	
					else
						%disp('else');
						dist(dist_i, dist_j)=old_dist(dist_i, dist_j) * exp((alpha1(t)));
						%if isnan(dist(dist_i, dist_j))
						%	disp(alpha1(t));
						%end
					end
				%end
			end
		end
		%disp(dist);
		
		sum_dist = sum(sum(dist));		
		dist = dist ./ sum_dist;
		if sum_dist==0
			disp(dist);
		end
	end
	%disp(alpha1);
	%disp(best_kernel);
	%return;
	O = zeros(size(h{1}));
	for t = 1:T
		O = O + alpha1(t) .* h{t};
	end	
end

function [new_y_train] = sample_examples(dist, y_train, n)	
	%disp(dist);
	dist_vec = reshape(dist, [1, size(dist,1)*size(dist,2)]);
	y_train_vec = reshape(y_train, [1, size(y_train,1)*size(y_train,2)]);
	[output, output_indices] = datasample(y_train_vec, n, 'Weights', dist_vec, 'Replace', false);
	new_y_train = zeros(size(y_train));
	new_y_train(output_indices)=y_train(output_indices);
end

%computes absolute loss
function [loss, predicted] = getLoss_1(predicted, original, num_ones, dist)
	%predicted = predicted .* dist;
	%disp(predicted);
	for i=1:size(predicted,1)
		for j=1:size(predicted,2)
			if predicted(i,j)>0.12
				predicted(i,j)=1;
			else
				predicted(i,j)=0;
			end
		end
	end
	%loss_matrix = abs(original-predicted); %.* dist;
	loss = 0;
	for i=1:size(predicted,1)
		for j=1:size(predicted,2)
			if original(i,j)==1 && predicted(i,j)~=original(i,j) 
				%if(~isreal(dist(i,j)))
				%	disp(dist(i,j)); return;
				%end
				%loss = loss + dist(i,j);
				loss = loss + 1;
			end	
		end
	end	
	number_of_ones = 0;
	for row=1:size(original,1)
		for col=1:size(original,2)
			if original(row,col)==1
				number_of_ones=number_of_ones+1;
			end
		end
	end
	if number_of_ones==0
		disp('Error');
		disp(original);
	end
	loss = loss/sum(sum(original==1));
	if (loss==1)
		loss = loss - 0.00001;
		%disp(loss);
	end
	if (loss==0)
		loss = loss + 0.00001;
		%disp(loss);
	end
	%disp('loss : '); disp(loss);
end

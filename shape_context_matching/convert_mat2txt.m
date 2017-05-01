Data = load('digit_100_train_hard.mat');
DataField = fieldnames(Data);
dlmwrite('digit_hard.txt', Data.(DataField{1}));
clean:
	rm -f ?.csv ??.csv ???.csv *.save history_*.png r2_*.csv keras_model*.h5 xgboost_model*.h5
	rm -rf ag_models_* keras_model_*.tf

clear:
	make clean

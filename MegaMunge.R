#### Data Tidying
Chuck_grip <- read_csv("TestingData/Chuck Grip.csv")
chuck_tidy <- Chuck_grip %>%
  select(rawDataOut1:rawDataOut8) %>% 
  rownames_to_column(var = "Electrode") %>% 
  gather(Electrode, Readout)
# View(chuck_tidy)


#### Data Visualization
ggplot(chuck_tidy, aes(x = chuck_tidy$Electrode, y=Readout, color = Readout))+ 
  geom_jitter() +xlab("Electrode") + ylab("Some fucking unit") + ggtitle("Signal Intensity")

                                          
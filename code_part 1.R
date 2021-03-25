options(scipen=999)
library(MASS)
library(fmsb)
library(car)
library(lme4)
library(rstatix)
library(pheatmap)
library(reshape2)
library(plyr)
library(dplyr)
library(ggplot2)
library(arm)
library(RColorBrewer)
library(cluster)
library(tidyverse)
library(dendextend)
library(factoextra)
library(sf)
library(ape)
library(dgof)

##################### cluster Analysis #################

### read data
all = read.csv('origin_traveller_all.csv')



### hierarchical clustering (all)
all_sub <- all[c("inc.20k_x","inc20.50k_x","inc50.100k_x","inc.100k_x" ,"pct_hbw_x",
                 "pct_hbo_x","pct_nhb_x","white_x","non.white_x",
                 "inc.20k_y","inc20.50k_y","inc50.100k_y","inc.100k_y" ,"pct_hbw_y",    
                 "pct_hbo_y","pct_nhb_y","white_y","non.white_y")]

all_sub %>%
  mutate_if(is.numeric, scale)
all_sub %>%
  mutate_if(is.numeric, round, digits = 2)
d = dist(all_sub,method="manhattan")

pheatmap(all_sub,show_rownames=FALSE,show_colnames=TRUE,
         clustering_method="complete",clustering_distance_cols="manhattan")

hcl=hclust(d, method = "complete")
plot(hcl,labels = FALSE, hang= -1, x_axis=TRUE)
rect.hclust(hcl, k=3, border = "red")
cluster=cutree(hcl,k=3)
all$cluster <- cluster
head(all)
fviz_nbclust(all_sub, FUN = hcut, method = "wss")
fviz_nbclust(all_sub, FUN = hcut, method = "silhouette")
gap_stat <- clusGap(all_sub, FUN = hcut, nstart = 25, K.max = 7, B = 25)
fviz_gap_stat(gap_stat)
#write.csv(all,'cluster_all.csv')



##################### KS tests #################

before = read.csv("D:/SI model/result/before_local_params.csv")
after = read.csv("D:/SI model/result/after_local_params.csv")


##### joining flow data
flow1 = read.csv("D:/SI model/result/flow_before_cluster.csv")
flow2 = read.csv("D:/SI model/result/flow_after_cluster.csv")
flow1 = flow1 %>% select(Origin, Destinatio, Flow, Duration, distance, 
                         Other_Serv,Recreation,Profession,Health_Car,
                         Retail,Accomodati,Finance, Real_Estat, Education)
flow2 = flow2 %>% select(Origin, Destinatio, Flow, Duration, distance, 
                         Other_Serv,Recreation,Profession,Health_Car,
                         Retail,Accomodati,Finance, Real_Estat, Education)
dat1 = before%>% select(ID, Cluster, Time, Other_Services, Recreation,
                        Professional, Health_Care, Retail, Accomodation_and_Food, 
                        Finance, Education, Real.Estate,Avg_trip_1,
                        pval_services_sig, pval_recreation_sig, 
                        pval_professional_sig, pval_health_sig, pval_retail_sig, 
                        pval_food_sig, pval_finance_sig, pval_edu_sig,
                        pval_trip_len_sig)
dat2 = after%>% select(ID, Cluster, Time, Other_Services, Recreation,
                       Professional, Health_Care, Retail, Accomodation_and_Food, 
                       Finance, Education, Real.Estate,Avg_trip_1,
                       pval_services_sig, pval_recreation_sig, 
                       pval_professional_sig, pval_health_sig, pval_retail_sig, 
                       pval_food_sig, pval_finance_sig, pval_edu_sig,
                       pval_trip_len_sig)
flow_dat1 = merge(flow1, dat1, by.x = c("Origin"), by.y = c("ID"), all = TRUE)
flow_dat2 = merge(flow2, dat2, by.x = c("Origin"), by.y = c("ID"), all = TRUE)

flow_dat = rbind(flow_dat2, flow_dat1)

####distance decay and attraction function
flow_dat$beta = -1*(flow_dat$Avg_trip_1)
flow_dat$dis_dcay = 1/(exp((flow_dat$Duration) * (flow_dat$beta))) 
flow_dat$att_service = (flow_dat$Other_Serv) ^ (flow_dat$Other_Services)
flow_dat$att_recreation = (flow_dat$Recreation.x) ^ (flow_dat$Recreation.y)
flow_dat$att_profession = (flow_dat$Profession) ^ (flow_dat$Professional)
flow_dat$att_health = (flow_dat$Health_Car) ^ (flow_dat$Health_Care)
flow_dat$att_retail = (flow_dat$Retail.x) ^ (flow_dat$Retail.y)
flow_dat$att_accomo = (flow_dat$Accomodati) ^ (flow_dat$Accomodation_and_Food)
flow_dat$att_fin = (flow_dat$Finance.x) ^ (flow_dat$Finance.y)
flow_dat$att_realest = (flow_dat$Real_Estat) ^ (flow_dat$Real.Estate)
flow_dat$Cluster = as.factor(flow_dat$Cluster)
flow_dat$att_edu = (flow_dat$Education.x) ^ (flow_dat$Education.y)

#### KS tests
high_bef = flow_dat[flow_dat$Time == "before" & flow_dat$Cluster == 'High SES',]
high_af = flow_dat[flow_dat$Time == "after" & flow_dat$Cluster == 'High SES',]
mod_bef = flow_dat[flow_dat$Time == "before" & flow_dat$Cluster == 'Moderate SES',]
mod_af = flow_dat[flow_dat$Time == "after" & flow_dat$Cluster == 'Moderate SES',]
low_bef = flow_dat[flow_dat$Time == "before" & flow_dat$Cluster == 'Low SES',]
low_af = flow_dat[flow_dat$Time == "after" & flow_dat$Cluster == 'Low SES',]


data1 = low_af
data2 = mod_af

a = ks.test(data2$dis_dcay, data1$dis_dcay, alternative = "two.sided")
b = ks.test(data2$att_profession, data1$att_profession, alternative = "two.sided")
c = ks.test(data2$att_service, data1$att_service,  alternative = "two.sided")
d = ks.test(data2$att_realest, data1$att_realest, alternative = "two.sided")
e = ks.test(data2$att_accomo, data1$att_accomo, alternative = "two.sided")
f = ks.test(data2$att_recreation, data1$att_recreation, alternative = "two.sided")
g = ks.test(data2$att_health, data1$att_health, alternative = "two.sided")
h = ks.test(data2$att_edu, data1$att_edu, alternative = "two.sided")
i = ks.test(data2$att_retail, data1$att_retail, alternative = "two.sided")
j = ks.test(data2$att_fin, data1$att_fin, alternative = "two.sided")

var = c("Duration", "Professional Serviced", "Service Jobs", "Rental and Leasing",
        "Accomodation and Food", "Recreational Facilities", "Health Care", "Education",
        "Retail", "Finance")
effect = data.frame(var)
D = c(a[[1]], b[[1]], c[[1]], d[[1]], e[[1]], f[[1]], 
      g[[1]], h[[1]], i[[1]], j[[1]])
p = c(a[[2]], b[[2]], c[[2]], d[[2]], e[[2]], f[[2]], 
      g[[2]], h[[2]], i[[2]], j[[2]])

effect$high_ses_D = D
effect$high_ses_p = p
effect$mod_ses_D = D
effect$mod_ses_p = p
effect$low_ses_D = D
effect$low_ses_p = p

effect$low_high_bef_D = D
effect$low_high_bef_p = p
effect$mod_high_bef_D = D
effect$mod_high_bef_p = p
effect$low_mod_bef_D = D
effect$low_mod_bef_p = p

effect$low_high_af_D = D
effect$low_high_af_p = p
effect$mod_high_af_D = D
effect$mod_high_af_p = p
effect$low_mod_af_D = D
effect$low_mod_af_p = p

write.csv(effect, "D:/SI model/result/effect.csv")

### distance decay
dist = subset(flow_dat, select= c('Origin','Flow','Duration', 'dis_dcay', 'Avg_trip_1',
                                  'pval_trip_len_sig', 'Cluster', 'Time'))
dist = dist[dist$Duration>1,]
dist = dist[dist$Flow>1,]
dist = as.data.frame(lapply(dist, rep, dist$Flow))
dist = arrange(dist, dis_dcay)
dist$Cluster = as.factor(dist$Cluster)

ggplot(dist, aes(x = dis_dcay, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0,1) +
  xlab("Distance Decay Effect") +
  ylab("Percent of trips") +
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))


### Service Facility
service = flow_dat %>% select(Flow,Other_Serv, att_service, 
                              pval_services_sig, Cluster, Time)
service = service[service$Other_Serv>1,]
service = service[service$Flow>1,]
service = as.data.frame(lapply(service, rep, service$Flow))

ggplot(service, aes(x = att_service, color = Cluster, linetype = Time)) +
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75,1.25) +
  xlab("Attractiveness Factor (Service Jobs)") +
  ylab("Percent of trips") + 
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))



### recreation 
recreation = flow_dat %>% select(Flow,Recreation.x, att_recreation, 
                                 pval_recreation_sig, Cluster, Time)
recreation = recreation[recreation$Recreation.x>1,]
recreation = recreation[recreation$Flow>1,]
recreation = as.data.frame(lapply(recreation, rep, recreation$Flow))
ggplot(recreation, aes(x = att_recreation, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Recreational Facilities)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))

### professional 
prof = flow_dat %>% select(Flow,Profession, att_profession,Cluster, Time)
prof = prof[prof$Profession>1,]
prof = prof[prof$Flow>1,]
prof = as.data.frame(lapply(prof, rep, prof$Flow))

ggplot(prof, aes(x = att_profession, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Professional Services)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))


### Real estate
realest = flow_dat %>% select(Flow,Real_Estat, att_realest,Cluster, Time)
realest = realest[realest$Real_Estat>1,]
realest = realest[realest$Flow>1,]
realest = as.data.frame(lapply(realest, rep, realest$Flow))

ggplot(realest, aes(x = att_realest, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Rental and Leasing)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical" , text = element_text(size = 45))


### Health
health = flow_dat %>% select(Flow,Health_Car, att_health,Cluster, Time)
health  = health [health $Health_Car>1,]
health  = health [health $Flow>1,]
health  = as.data.frame(lapply(health , rep, health $Flow))

ggplot(health , aes(x = att_health, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Health Care)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))


### Accomodation and food
accomo = flow_dat %>% select(Flow,Accomodati, att_accomo,Cluster, Time)
accomo  = accomo[accomo$Accomodati>1,]
accomo  = accomo[accomo$Flow>1,]
accomo  = as.data.frame(lapply(accomo, rep, accomo$Flow))

ggplot(accomo, aes(x = att_accomo, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Accomodation and Food)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))

### Education
edu = flow_dat %>% select(Flow,Education.x, att_edu,Cluster, Time)
edu  = edu[edu$Education.x>1,]
edu  = edu[edu$Flow>1,]
edu  = as.data.frame(lapply(edu, rep, edu$Flow))

ggplot(edu, aes(x = att_edu, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Education)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))

### Retail
retail = flow_dat %>% select(Flow,Retail.x, att_retail,Cluster, Time)
retail  = retail[retail$Retail.x>1,]
retail  = retail[retail$Flow>1,]
retail  = as.data.frame(lapply(retail, rep, retail$Flow))

ggplot(retail, aes(x = att_retail, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Retail)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))

### finance
fin = flow_dat %>% select(Flow,Finance.x, att_fin,Cluster, Time)
fin  = fin[fin$Finance.x>1,]
fin  = fin[fin$Flow>1,]
fin  = as.data.frame(lapply(fin, rep, fin$Flow))

ggplot(fin, aes(x = att_fin, color = Cluster, linetype = Time)) + 
  stat_ecdf(geom = "line", size =3) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() + xlim(0.75, 1.25) +
  xlab("Attractiveness Factor (Finance)") +
  ylab("Percent of trips")+
  scale_color_manual(values=c("#2c7bb6","#d7191c", "#fc8d59")) +
  theme(legend.position="bottom", legend.box="vertical", text = element_text(size = 45))

